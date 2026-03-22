"""
Offline benchmarks for paper-aligned components that do not need a running HTTP server:

- Monte Carlo dropout: T=0 vs T>0 variance + Table-1 depth proxy on fixed strings
- DSRG: same evidence query with use_dsrg on/off (score / reliability deltas)
- Multi-modal fusion: synthetic CLS + image/social/temporal → fused fake % + attention

Requires a veracity checkpoint (baseline, DANN, or DIML).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(SCRIPTS))
import core.torch_env  # noqa: E402, F401

import torch
from transformers import AutoTokenizer

from device_util import device_pretty, resolve_device  # noqa: E402
from evaluate import load_model  # noqa: E402
from features.routing.mc_uncertainty import (  # noqa: E402
    predict_with_mc_dropout,
    table1_depth_from_fake_variance,
)
from features.multimodal_fusion.fusion import (  # noqa: E402
    fuse_detection_signals,
    get_multimodal_fusion,
)
from features.evidence_retrieval_3.dsrg import build_dsrg_from_kb_facts  # noqa: E402
from features.evidence_retrieval_3.retriever import ExpandedKnowledgeBase  # noqa: E402


PROBE_TEXTS = [
    "Breaking: experts say the vaccine rollout exceeded expectations in major cities.",
    "Local diner wins pie contest for third year amid heavy rain downtown.",
]


def mc_section(model, tokenizer, device: torch.device, t_mc: int) -> dict:
    rows = []
    for text in PROBE_TEXTS:
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        t0 = time.perf_counter()
        r0 = predict_with_mc_dropout(model, ids, mask, T=0)
        ms0 = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        r1 = predict_with_mc_dropout(model, ids, mask, T=t_mc)
        ms1 = (time.perf_counter() - t1) * 1000.0
        rows.append(
            {
                "text_preview": text[:80],
                "T0": {
                    "mean_fake": r0["mean_fake"],
                    "var_fake": r0["var_fake"],
                    "latency_ms": round(ms0, 3),
                    "table1_depth": table1_depth_from_fake_variance(r0["var_fake"]),
                },
                f"T{t_mc}": {
                    "mean_fake": r1["mean_fake"],
                    "var_fake": r1["var_fake"],
                    "latency_ms": round(ms1, 3),
                    "table1_depth": table1_depth_from_fake_variance(r1["var_fake"]),
                },
            }
        )
    return {
        "task": "mc_dropout_routing_probe",
        "T_mc": t_mc,
        "samples": rows,
        "note": "Forward latency grows ~linearly with T; routing uses var_fake when T>0.",
    }


def dsrg_section() -> dict:
    """
    Lightweight graph stats only (no full LIAR load / FAISS). For end-to-end retrieval
    scores, call POST /evidence with use_dsrg on a running API.
    """
    t0 = time.perf_counter()
    kb = ExpandedKnowledgeBase(load_datasets=False)
    g = build_dsrg_from_kb_facts(kb.facts)
    ms = (time.perf_counter() - t0) * 1000.0
    high = "World Health Organization"
    low = "Unknown"
    return {
        "task": "dsrg_graph_probe",
        "kb_facts_used": len(kb.facts),
        "graph_nodes": len(g._idx_to_source),
        "build_latency_ms": round(ms, 3),
        "reliability_sample_high_trust": g.reliability(high),
        "reliability_sample_unknown": g.reliability(low),
        "boost_score_at_0_8_relevance": {
            "high_trust_source": g.boost_score(0.8, high),
            "unknown_source": g.boost_score(0.8, low),
        },
        "note": "Hand-crafted KB only; full API retrieval adds LIAR + semantic search.",
    }


def multimodal_section(device: torch.device) -> dict:
    torch.manual_seed(42)
    text_cls = torch.randn(1, 768)
    fus = get_multimodal_fusion(device="cpu")
    fused, dbg = fuse_detection_signals(
        fus,
        text_cls,
        58.0,
        72.0,
        35.0,
        {"likes": 8000.0, "shares": 120.0, "comments": 400.0, "account_verified": False},
        "2023-06-10T08:30:00+00:00",
        device,
    )
    return {
        "task": "multimodal_fusion_probe",
        "synthetic": True,
        "fused_fake_probability_percent": fused,
        "fusion_debug": dbg,
        "note": "Synthetic CLS embedding; use /detect for end-to-end numbers on real inputs.",
    }


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Veracity checkpoint (baseline / DANN / DIML)",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--mc_T", type=int, default=10, help="MC passes for probe (paper often uses 30)")
    ap.add_argument("--skip_mc", action="store_true")
    ap.add_argument("--skip_dsrg", action="store_true")
    ap.add_argument("--skip_multimodal", action="store_true")
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    out: dict = {"task": "paper_components_bundle", "sections": {}}

    mp = Path(args.model_path) if args.model_path else None
    if not args.skip_mc:
        if not mp or not mp.exists():
            out["sections"]["mc_dropout"] = {"ok": False, "error": "missing --model_path"}
        else:
            try:
                device = resolve_device(args.device)
                # MPS: avoid state_dict copy glitches; MC probe is small on CPU
                run_dev = torch.device("cpu") if device.type == "mps" else device
                model = load_model(mp, run_dev)
                tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
                out["sections"]["mc_dropout"] = {
                    "ok": True,
                    "device": device_pretty(run_dev),
                    "requested_device": device_pretty(device),
                    "data": mc_section(model, tokenizer, run_dev, max(1, int(args.mc_T))),
                }
            except BaseException as e:
                out["sections"]["mc_dropout"] = {
                    "ok": False,
                    "error": repr(e),
                    "hint": "Retry with --device cpu or a smaller checkpoint; OOM/MPS issues are common on laptops.",
                }

    if not args.skip_dsrg:
        try:
            out["sections"]["dsrg"] = {"ok": True, "data": dsrg_section()}
        except Exception as e:
            out["sections"]["dsrg"] = {"ok": False, "error": repr(e)}

    if not args.skip_multimodal:
        try:
            dev = torch.device("cpu")
            out["sections"]["multimodal_fusion"] = {"ok": True, "data": multimodal_section(dev)}
        except Exception as e:
            out["sections"]["multimodal_fusion"] = {"ok": False, "error": repr(e)}

    outp = Path(args.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {outp.resolve()}")


if __name__ == "__main__":
    main()
