#!/usr/bin/env python3
"""
CPU quick demo: tiny subsets, 1 epoch, baseline veracity + stance + short eval.
Typical wall time: a few minutes (first run longer: Hugging Face model download).

From repository root:
  python3 training/scripts/quick_demo_cpu.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "training/scripts/orchestrate.py"),
        "--quick-demo",
    ]
    # Forward extra args (e.g. --quick-demo-domain) to orchestrate
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
