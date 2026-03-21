"""
Measure mean wall time for TextClassifier forward passes (batch size 1), optional CUDA sync.
"""

from __future__ import annotations

import argparse
import json
import statistics
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
from train_text_model import TextClassifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n_runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    mp = Path(args.model_path)
    if not mp.exists():
        raise SystemExit(f"Missing model: {mp}")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    text = (
        "Breaking: officials announced new measures after reports surfaced online. "
        "Critics questioned the timing and sourcing of the claims."
    )
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    ckpt = torch.load(mp, map_location=device)
    model = TextClassifier()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    use_cuda = device.type == "cuda"

    @torch.no_grad()
    def forward_once():
        model(input_ids, attention_mask)

    for _ in range(args.warmup):
        forward_once()
        if use_cuda:
            torch.cuda.synchronize()

    times = []
    for _ in range(args.n_runs):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward_once()
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    ms_mean = statistics.mean(times)
    ms_p50 = statistics.median(times)
    ms_p95 = sorted(times)[int(0.95 * (len(times) - 1))] if len(times) > 1 else times[0]

    out = {
        "task": "veracity_single_forward_latency",
        "device": device_pretty(device),
        "model_path": str(mp.resolve()),
        "n_runs": args.n_runs,
        "batch_size": 1,
        "mean_ms": round(ms_mean, 4),
        "p50_ms": round(ms_p50, 4),
        "p95_ms": round(ms_p95, 4),
        "note": "Text encoder only; not full /detect API pipeline.",
    }
    print(json.dumps(out, indent=2))

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {outp.resolve()}")


if __name__ == "__main__":
    main()
