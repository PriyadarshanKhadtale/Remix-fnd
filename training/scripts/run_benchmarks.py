#!/usr/bin/env python3
"""
Train DistilRoBERTa veracity classifier on FakeNewsNet JSON splits, then run full test eval.

Designed for Google Colab (T4) and local GPU/MPS. From repository root:

  python3 training/scripts/run_benchmarks.py --device auto
  python3 training/scripts/run_benchmarks.py --device cuda --epochs 3

Writes benchmark_summary.json with train args + eval metrics paths.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n", "=" * 60, "\n", " ".join(cmd), "\n", "=" * 60, flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="REMIX-FND text benchmark: train + full eval")
    ap.add_argument("--data_dir", type=str, default="data/processed/fakenewsnet")
    ap.add_argument("--output_dir", type=str, default="models/text_classifier")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | cpu | mps",
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap train/val/test rows (smoke test only)",
    )
    ap.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training; only evaluate existing best_model.pt",
    )
    ap.add_argument(
        "--eval_json",
        type=str,
        default="benchmark_eval_metrics.json",
        help="Path for evaluate.py --output_json (under repo root unless absolute)",
    )
    ap.add_argument(
        "--summary_json",
        type=str,
        default="benchmark_summary.json",
        help="Aggregate summary written by this script",
    )
    args = ap.parse_args()

    py = sys.executable
    data_dir = ROOT / args.data_dir
    out_model_dir = ROOT / args.output_dir
    test_json = data_dir / "test.json"
    eval_json_path = Path(args.eval_json)
    if not eval_json_path.is_absolute():
        eval_json_path = ROOT / eval_json_path
    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    if not test_json.exists():
        raise SystemExit(f"Missing test data: {test_json}")

    if not args.skip_train:
        cmd_t = [
            py,
            str(SCRIPTS / "train_text_model.py"),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(out_model_dir),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.train_batch_size),
            "--lr",
            str(args.lr),
            "--device",
            args.device,
        ]
        if args.max_samples is not None:
            cmd_t.extend(["--max_samples", str(args.max_samples)])
        run(cmd_t)
    else:
        ckpt = out_model_dir / "best_model.pt"
        if not ckpt.exists():
            raise SystemExit(f"--skip_train but missing checkpoint: {ckpt}")

    ckpt = out_model_dir / "best_model.pt"
    cmd_e = [
        py,
        str(SCRIPTS / "evaluate.py"),
        "--model_path",
        str(ckpt),
        "--test_data",
        str(test_json),
        "--batch_size",
        str(args.eval_batch_size),
        "--device",
        args.device,
        "--output_json",
        str(eval_json_path),
    ]
    if args.max_samples is not None:
        cmd_e.extend(["--max_samples", str(args.max_samples)])
    run(cmd_e)

    eval_metrics = json.loads(eval_json_path.read_text(encoding="utf-8"))
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_flag": args.device,
        "train": {
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "lr": args.lr,
            "skipped": args.skip_train,
            "max_samples": args.max_samples,
        },
        "eval_json": str(eval_json_path.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "metrics": {
            "accuracy_percent": eval_metrics.get("accuracy_percent"),
            "macro_f1": eval_metrics.get("macro_f1"),
            "macro_precision": eval_metrics.get("macro_precision"),
            "macro_recall": eval_metrics.get("macro_recall"),
            "n_test": eval_metrics.get("n_test"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n✅ Benchmark pipeline finished.")
    print(f"   Eval metrics: {eval_json_path.resolve()}")
    print(f"   Summary:      {summary_path.resolve()}")
    print(
        f"   Accuracy %: {summary['metrics']['accuracy_percent']}, "
        f"macro F1: {summary['metrics']['macro_f1']}"
    )


if __name__ == "__main__":
    main()
