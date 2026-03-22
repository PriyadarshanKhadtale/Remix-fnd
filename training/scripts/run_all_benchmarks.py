#!/usr/bin/env python3
"""
Run all automated benchmarks in the repo (train + eval where applicable).

Outputs under --run_dir (default: benchmark_runs/<UTC timestamp>):
  manifest.json              — master index + paper-scope gaps
  veracity_baseline.*        — train + sklearn metrics JSON
  veracity_dann.* (or diml)  — domain-adversarial or DIML train + test metrics
  stance_liar_test.json      — stance model on LIAR test.tsv
  ai_ensemble_micro.json     — fixed-string AI detector smoke
  veracity_latency.json      — single-forward timing (text encoder only)
  ood_domains.json           — per-domain + overall macro F1 (evaluate_ood_domains.py)
  paper_components_probe.json — MC dropout, DSRG, multimodal fusion (offline probes)

One command (full):
  python3 training/scripts/run_all_benchmarks.py --device auto

Quick smoke:
  python3 training/scripts/run_all_benchmarks.py --quick --device cpu

Skip heavy pieces:
  --skip_ood --skip_paper_components --skip_dann ...
Use DIML instead of DANN for step 2:
  --diml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent

PAPER_GAPS = [
    {
        "item": "multi_dataset_ood_table",
        "status": "not_automated",
        "detail": "Manuscript OOD / six-corpus Table 5-style eval requires frozen splits in config.yaml and trained full stack.",
    },
    {
        "item": "full_api_latency",
        "status": "partial",
        "detail": "Script benchmarks text-encoder forward only; full /detect pipeline needs separate HTTP harness + warm instances.",
    },
    {
        "item": "human_study",
        "status": "not_automated",
        "detail": "IRB-gated; no instrument in repo.",
    },
    {
        "item": "paraphrase_attack",
        "status": "not_automated",
        "detail": "Requires frozen paraphrase model + attack loop (not in training/scripts).",
    },
]


def run_step(
    manifest: dict,
    name: str,
    cmd: list[str],
    py: str,
) -> bool:
    print("\n" + "=" * 72 + f"\n STEP: {name}\n" + "=" * 72 + "\n" + " ".join(cmd) + "\n", flush=True)
    entry = {"name": name, "command": " ".join(cmd), "ok": False, "artifacts": []}
    manifest["steps"].append(entry)
    try:
        r = subprocess.run(cmd, cwd=str(ROOT))
        entry["ok"] = r.returncode == 0
        return r.returncode == 0
    except Exception as e:
        entry["error"] = repr(e)
        entry["traceback"] = traceback.format_exc()
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Output directory (default: benchmark_runs/<iso-time>)",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs_veracity", type=int, default=3)
    ap.add_argument("--epochs_dann", type=int, default=2)
    ap.add_argument("--epochs_stance", type=int, default=2)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: 1 epoch, capped samples, smaller batches",
    )
    ap.add_argument("--skip_dann", action="store_true")
    ap.add_argument("--skip_stance", action="store_true")
    ap.add_argument("--skip_ai_micro", action="store_true")
    ap.add_argument("--skip_latency", action="store_true")
    ap.add_argument(
        "--skip_ood",
        action="store_true",
        help="Skip evaluate_ood_domains.py (per-domain F1 on test.json)",
    )
    ap.add_argument(
        "--skip_paper_components",
        action="store_true",
        help="Skip paper_components_probe.json (MC dropout, DSRG, multimodal offline)",
    )
    ap.add_argument(
        "--diml",
        action="store_true",
        help="Train DIML (train_diml.py) instead of DANN for the domain-adversarial slot",
    )
    args = ap.parse_args()

    py = sys.executable
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "benchmark_runs" / ts
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    data_fn = ROOT / "data/processed/fakenewsnet"
    test_json = data_fn / "test.json"

    q = args.quick
    ep_v = 1 if q else args.epochs_veracity
    ep_d = 1 if q else args.epochs_dann
    ep_s = 1 if q else args.epochs_stance
    tb = 8 if q else args.train_batch_size
    eb = 16 if q else args.eval_batch_size
    max_v = 2048 if q else None
    max_st = 4000 if q else None
    max_st_eval = 2000 if q else None

    out_baseline = run_dir / "baseline_veracity"
    out_dann = run_dir / ("diml_veracity" if args.diml else "dann_veracity")
    out_stance = run_dir / "stance"
    out_baseline.mkdir(parents=True, exist_ok=True)
    out_dann.mkdir(parents=True, exist_ok=True)
    out_stance.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "device_flag": args.device,
        "quick": q,
        "steps": [],
        "paper_gaps": PAPER_GAPS,
    }

    ok_all = True

    # 1) Baseline veracity
    cmd1 = [
        py,
        str(SCRIPTS / "train_text_model.py"),
        "--data_dir",
        str(data_fn),
        "--output_dir",
        str(out_baseline),
        "--epochs",
        str(ep_v),
        "--batch_size",
        str(tb),
        "--device",
        args.device,
    ]
    if max_v:
        cmd1.extend(["--max_samples", str(max_v)])
    if not run_step(manifest, "train_veracity_baseline", cmd1, py):
        ok_all = False

    j1 = run_dir / "veracity_baseline_eval.json"
    cmd1e = [
        py,
        str(SCRIPTS / "evaluate.py"),
        "--model_path",
        str(out_baseline / "best_model.pt"),
        "--test_data",
        str(test_json),
        "--batch_size",
        str(eb),
        "--device",
        args.device,
        "--output_json",
        str(j1),
    ]
    if max_v:
        cmd1e.extend(["--max_samples", str(max_v)])
    if (out_baseline / "best_model.pt").exists():
        if run_step(manifest, "eval_veracity_baseline", cmd1e, py):
            manifest["steps"][-1]["artifacts"].append(str(j1))
        else:
            ok_all = False
    else:
        manifest["steps"].append(
            {
                "name": "eval_veracity_baseline",
                "ok": False,
                "skipped": "missing checkpoint",
            }
        )
        ok_all = False

    # 2) DANN or DIML veracity
    if not args.skip_dann:
        train_script = "train_diml.py" if args.diml else "train_domain_adversarial.py"
        step_train = "train_veracity_diml" if args.diml else "train_veracity_dann"
        cmd2 = [
            py,
            str(SCRIPTS / train_script),
            "--data_dir",
            str(data_fn),
            "--output_dir",
            str(out_dann),
            "--epochs",
            str(ep_d),
            "--batch_size",
            str(tb),
            "--device",
            args.device,
        ]
        if max_v:
            cmd2.extend(["--max_samples", str(max_v)])
        if q and args.diml:
            cmd2.extend(
                [
                    "--tasks_per_step",
                    "1",
                    "--support_n",
                    "4",
                    "--query_n",
                    "4",
                    "--inner_steps",
                    "2",
                ]
            )
        if not run_step(manifest, step_train, cmd2, py):
            ok_all = False

        j2 = run_dir / ("veracity_diml_eval.json" if args.diml else "veracity_dann_eval.json")
        eval_step = "eval_veracity_diml" if args.diml else "eval_veracity_dann"
        cmd2e = [
            py,
            str(SCRIPTS / "evaluate.py"),
            "--model_path",
            str(out_dann / "best_model.pt"),
            "--test_data",
            str(test_json),
            "--batch_size",
            str(eb),
            "--device",
            args.device,
            "--output_json",
            str(j2),
        ]
        if max_v:
            cmd2e.extend(["--max_samples", str(max_v)])
        if (out_dann / "best_model.pt").exists():
            if run_step(manifest, eval_step, cmd2e, py):
                manifest["steps"][-1]["artifacts"].append(str(j2))
            else:
                ok_all = False
        else:
            manifest["steps"].append(
                {"name": eval_step, "ok": False, "skipped": "missing checkpoint"}
            )
            ok_all = False
    else:
        manifest["steps"].append({"name": "dann_or_diml_pipeline", "ok": True, "skipped": True})

    # 3) Stance
    if not args.skip_stance:
        cmd3 = [
            py,
            str(SCRIPTS / "train_stance_cross_encoder.py"),
            "--output_dir",
            str(out_stance),
            "--epochs",
            str(ep_s),
            "--batch_size",
            str(tb),
            "--device",
            args.device,
        ]
        if max_st:
            cmd3.extend(["--max_samples", str(max_st)])
        if not run_step(manifest, "train_stance_liar", cmd3, py):
            ok_all = False

        j3 = run_dir / "stance_liar_test.json"
        cmd3e = [
            py,
            str(SCRIPTS / "evaluate_stance_liar.py"),
            "--checkpoint",
            str(out_stance / "best_model.pt"),
            "--liar_split",
            "test",
            "--batch_size",
            str(eb),
            "--device",
            args.device,
            "--output_json",
            str(j3),
        ]
        if max_st_eval:
            cmd3e.extend(["--max_samples", str(max_st_eval)])
        if (out_stance / "best_model.pt").exists():
            if run_step(manifest, "eval_stance_liar_test", cmd3e, py):
                manifest["steps"][-1]["artifacts"].append(str(j3))
            else:
                ok_all = False
        else:
            manifest["steps"].append(
                {"name": "eval_stance_liar_test", "ok": False, "skipped": "missing checkpoint"}
            )
            ok_all = False
    else:
        manifest["steps"].append({"name": "stance_pipeline", "ok": True, "skipped": True})

    # 4) AI ensemble micro
    if not args.skip_ai_micro:
        j4 = run_dir / "ai_ensemble_micro.json"
        cmd4 = [
            py,
            str(SCRIPTS / "benchmark_ai_ensemble.py"),
            "--output_json",
            str(j4),
        ]
        if run_step(manifest, "ai_ensemble_micro", cmd4, py):
            manifest["steps"][-1]["artifacts"].append(str(j4))
        else:
            ok_all = False
    else:
        manifest["steps"].append({"name": "ai_ensemble_micro", "ok": True, "skipped": True})

    # 5) Latency (baseline checkpoint)
    if not args.skip_latency and (out_baseline / "best_model.pt").exists():
        j5 = run_dir / "veracity_latency.json"
        cmd5 = [
            py,
            str(SCRIPTS / "benchmark_veracity_latency.py"),
            "--model_path",
            str(out_baseline / "best_model.pt"),
            "--device",
            args.device,
            "--n_runs",
            "40" if q else "80",
            "--output_json",
            str(j5),
        ]
        if run_step(manifest, "veracity_forward_latency", cmd5, py):
            manifest["steps"][-1]["artifacts"].append(str(j5))
        else:
            ok_all = False
    else:
        manifest["steps"].append(
            {
                "name": "veracity_forward_latency",
                "ok": True,
                "skipped": args.skip_latency or not (out_baseline / "best_model.pt").exists(),
            }
        )

    # 6) OOD / per-domain evaluation (baseline checkpoint = in-domain reference)
    if not args.skip_ood and (out_baseline / "best_model.pt").exists():
        j_ood = run_dir / "ood_domains.json"
        cmd_ood = [
            py,
            str(SCRIPTS / "evaluate_ood_domains.py"),
            "--model_path",
            str(out_baseline / "best_model.pt"),
            "--test_data",
            str(test_json),
            "--batch_size",
            str(eb),
            "--device",
            args.device,
            "--output_json",
            str(j_ood),
        ]
        if max_v:
            cmd_ood.extend(["--max_samples", str(max_v)])
        if run_step(manifest, "eval_ood_domains", cmd_ood, py):
            manifest["steps"][-1]["artifacts"].append(str(j_ood))
        else:
            ok_all = False
    else:
        manifest["steps"].append(
            {
                "name": "eval_ood_domains",
                "ok": True,
                "skipped": args.skip_ood or not (out_baseline / "best_model.pt").exists(),
            }
        )

    # 7) MC dropout + DSRG + multimodal fusion (offline probes)
    if not args.skip_paper_components and (out_baseline / "best_model.pt").exists():
        j_pc = run_dir / "paper_components_probe.json"
        cmd_pc = [
            py,
            str(SCRIPTS / "benchmark_paper_components.py"),
            "--model_path",
            str(out_baseline / "best_model.pt"),
            "--device",
            args.device,
            "--output_json",
            str(j_pc),
        ]
        if q:
            cmd_pc.extend(["--mc_T", "5"])
        else:
            cmd_pc.extend(["--mc_T", "30"])
        if run_step(manifest, "paper_components_probe", cmd_pc, py):
            manifest["steps"][-1]["artifacts"].append(str(j_pc))
        else:
            ok_all = False
    else:
        manifest["steps"].append(
            {
                "name": "paper_components_probe",
                "ok": True,
                "skipped": args.skip_paper_components
                or not (out_baseline / "best_model.pt").exists(),
            }
        )

    manifest["all_ok"] = ok_all
    mf_path = run_dir / "manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print("MANIFEST:", mf_path)
    print(json.dumps(manifest, indent=2)[:8000])
    if len(json.dumps(manifest)) > 8000:
        print("... (truncated; see manifest.json)")
    print("=" * 72 + "\n")

    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
