"""
Run training scripts in order: veracity (optional DANN or DIML) → stance cross-encoder.

Usage (from repo root):
  python3 training/scripts/quick_demo_cpu.py
  python3 training/scripts/orchestrate.py --quick-demo
  python3 training/scripts/orchestrate.py --device cpu
  python3 training/scripts/orchestrate.py --skip-domain --device cuda
  python3 training/scripts/orchestrate.py --diml --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n", "=" * 60, "\n", " ".join(cmd), "\n", "=" * 60, flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Passed to training scripts: auto | cuda | cpu | mps",
    )
    ap.add_argument("--skip-domain", action="store_true", help="Train only baseline text model")
    ap.add_argument(
        "--diml",
        action="store_true",
        help="Train DIML (DANN + episodic MAML on heads) instead of DANN-only",
    )
    ap.add_argument("--skip-stance", action="store_true")
    ap.add_argument("--epochs-text", type=int, default=3)
    ap.add_argument("--epochs-domain", type=int, default=2)
    ap.add_argument("--epochs-diml", type=int, default=None, help="Epochs for DIML (default: epochs-domain)")
    ap.add_argument("--epochs-stance", type=int, default=2)
    ap.add_argument(
        "--quick-demo",
        action="store_true",
        help="CPU-friendly: 1 epoch, small data caps, batch 8; baseline veracity unless --quick-demo-domain",
    )
    ap.add_argument(
        "--quick-demo-domain",
        action="store_true",
        help="With --quick-demo, train domain-adversarial instead of baseline text",
    )
    args = ap.parse_args()

    py = sys.executable
    data_dir = ROOT / "data/processed/fakenewsnet"
    out_text = ROOT / "models/text_classifier"

    if args.quick_demo:
        args.device = "cpu"
        args.epochs_text = 1
        args.epochs_domain = 1
        args.epochs_stance = 1
        if args.quick_demo_domain:
            args.skip_domain = False
        else:
            args.skip_domain = True
        demo_n = 512
        demo_stance = 2000
        demo_eval = 400
        demo_batch = 8
        print(
            "\n*** QUICK DEMO (CPU): "
            f"max {demo_n} rows/split veracity, {demo_stance} stance pairs, "
            f"eval {demo_eval} test rows, batch {demo_batch} ***\n"
        )
    else:
        demo_n = demo_stance = demo_eval = demo_batch = None  # type: ignore

    if args.skip_domain:
        cmd_t = [
            py,
            str(SCRIPTS / "train_text_model.py"),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(out_text),
            "--epochs",
            str(args.epochs_text),
            "--device",
            args.device,
        ]
        if args.quick_demo:
            cmd_t.extend(
                ["--max_samples", str(demo_n), "--batch_size", str(demo_batch)]
            )
        run(cmd_t)
    else:
        ep_dom = args.epochs_diml if args.epochs_diml is not None else args.epochs_domain
        script = "train_diml.py" if args.diml else "train_domain_adversarial.py"
        cmd_d = [
            py,
            str(SCRIPTS / script),
            "--data_dir",
            str(data_dir),
            "--output_dir",
            str(out_text),
            "--epochs",
            str(ep_dom),
            "--device",
            args.device,
        ]
        if args.quick_demo:
            cmd_d.extend(
                ["--max_samples", str(demo_n), "--batch_size", str(demo_batch)]
            )
            if args.diml:
                cmd_d.extend(
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
        run(cmd_d)

    if not args.skip_stance:
        cmd_s = [
            py,
            str(SCRIPTS / "train_stance_cross_encoder.py"),
            "--epochs",
            str(args.epochs_stance),
            "--device",
            args.device,
        ]
        if args.quick_demo:
            cmd_s.extend(["--max_samples", str(demo_stance), "--batch_size", str(demo_batch)])
        run(cmd_s)

    cmd_e = [
        py,
        str(SCRIPTS / "evaluate.py"),
        "--model_path",
        str(out_text / "best_model.pt"),
        "--test_data",
        str(data_dir / "test.json"),
        "--device",
        args.device,
    ]
    if args.quick_demo:
        cmd_e.extend(["--max_samples", str(demo_eval), "--batch_size", str(demo_batch)])
    run(cmd_e)

    print(
        "\n✅ Pipeline finished.\n"
        "  • Veracity: models/text_classifier/best_model.pt (set REMIX_VERACITY_CKPT to override)\n"
        "  • Stance:   models/stance_cross_encoder/best_model.pt (auto-loaded by EvidenceRetriever)\n"
        "  • Start API: cd backend && python3 run.py\n"
    )


if __name__ == "__main__":
    main()
