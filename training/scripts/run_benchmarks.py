#!/usr/bin/env python3
"""
Reproducible veracity benchmarks for REMIX-FND (FakeNewsNet-style test.json).

From repository root:
  python3 training/scripts/run_benchmarks.py
  python3 training/scripts/run_benchmarks.py --device cuda --latency_runs 100

Writes benchmark_results/benchmark_<timestamp>.json and prints a summary.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def pick_device(preferred: Optional[str]) -> str:
    if preferred and preferred != "auto":
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run veracity classification benchmarks")
    ap.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT / "models/text_classifier/best_model.pt"),
    )
    ap.add_argument(
        "--test_data",
        type=str,
        default=str(ROOT / "data/processed/fakenewsnet/test.json"),
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | mps",
    )
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument(
        "--latency_runs",
        type=int,
        default=50,
        help="Forward passes (batch size 1) for mean inference latency; 0 to skip",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "benchmark_results"),
    )
    args = ap.parse_args()

    device_s = pick_device(args.device)
    model_path = Path(args.model_path)
    test_path = Path(args.test_data)

    if not model_path.is_file():
        print(
            f"Missing checkpoint: {model_path}\n"
            "Train first: python3 training/scripts/train_text_model.py "
            "--data_dir data/processed/fakenewsnet --output_dir models/text_classifier "
            f"--epochs 3 --device {device_s}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not test_path.is_file():
        print(f"Missing test data: {test_path}", file=sys.stderr)
        raise SystemExit(1)

    sys.path.insert(0, str(ROOT / "backend"))
    import core.torch_env  # noqa: F401, E402

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import (  # noqa: E402
        compute_metrics_bundle,
        load_model,
        run_eval_with_scores,
    )
    from train_text_model import NewsDataset  # noqa: E402

    device = torch.device(device_s)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    ds = NewsDataset(test_path, tokenizer, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    loader_lat = DataLoader(ds, batch_size=1, shuffle=False)

    model = load_model(model_path, device)

    t0 = time.perf_counter()
    y_true, y_pred, y_score = run_eval_with_scores(model, loader, device, collect_scores=True)
    eval_seconds = time.perf_counter() - t0

    metrics = compute_metrics_bundle(y_true, y_pred, y_score)

    latency_ms_mean: Optional[float] = None
    if args.latency_runs > 0:
        model.eval()
        it = iter(loader_lat)
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Warmup
        for _ in range(5):
            _ = model(input_ids, attention_mask)
        if device_s == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.latency_runs):
            _ = model(input_ids, attention_mask)
        if device_s == "cuda":
            torch.cuda.synchronize()
        latency_ms_mean = (time.perf_counter() - t1) * 1000.0 / max(args.latency_runs, 1)

    import torch as _torch

    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": _torch.__version__,
        "device_requested": args.device,
        "device_used": device_s,
        "model_path": str(model_path.resolve()),
        "test_data": str(test_path.resolve()),
        "n_examples": len(y_true),
        "batch_size_eval": args.batch_size,
        "full_eval_wall_seconds": round(eval_seconds, 3),
        "metrics": metrics,
        "latency_batch1_ms_mean": latency_ms_mean,
        "latency_runs": args.latency_runs if args.latency_runs > 0 else None,
        "note": (
            "Veracity head only on processed FakeNewsNet JSON; not end-to-end API / multi-modal. "
            "README latency (~400–500 ms) refers to full pipeline."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    latest = out_dir / "latest.json"
    latest.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(json.dumps(record, indent=2))
    print(f"\nSaved: {out_path}\nSaved: {latest}")


if __name__ == "__main__":
    main()
