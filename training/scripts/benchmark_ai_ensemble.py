"""
Micro-benchmark: run AIContentDetector (6-detector ensemble) on fixed short texts.
Not a substitute for HC3 leaderboard eval; records latency + scores for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

from features.ai_detection_4.detector import AIContentDetector  # noqa: E402


# Curated snippets long enough for detectors (>= ~30 chars where needed)
_MICRO_TEXTS = [
    {"id": "h1", "kind": "humanish", "text": (
        "lol no way — I was gonna grab coffee but tbh the line was insane. "
        "Whatever, I'll sorta wait... maybe?? idk imo that's kinda ridiculous haha!!!"
    )},
    {"id": "h2", "kind": "humanish", "text": (
        "Dunno if this rumor is true; my cousin said she heard it from someone at work. "
        "Could be nonsense. We should verify before sharing."
    )},
    {"id": "a1", "kind": "aiish", "text": (
        "The implementation leverages a multi-modal architecture to optimize downstream "
        "performance while maintaining robustness across heterogeneous data distributions. "
        "Furthermore, the methodology facilitates scalable deployment in production environments."
    )},
    {"id": "a2", "kind": "aiish", "text": (
        "In conclusion, it is important to note that various factors may contribute to "
        "the observed outcomes. Additional research is warranted to further elucidate "
        "the underlying mechanisms and validate the preliminary findings presented herein."
    )},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    det = AIContentDetector()
    for _ in range(args.warmup):
        det.detect(_MICRO_TEXTS[0]["text"])

    rows = []
    latencies_ms = []
    for item in _MICRO_TEXTS:
        t0 = time.perf_counter()
        r = det.detect(item["text"])
        ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(ms)
        rows.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "latency_ms": round(ms, 3),
                "probability": float(r.get("probability", 0.0)),
                "confidence": float(r.get("confidence", 0.0)),
                "is_ai_generated": bool(r.get("is_ai_generated")),
                "verdict": (r.get("verdict") or "")[:200],
            }
        )

    summary = {
        "task": "ai_ensemble_micro",
        "n_texts": len(_MICRO_TEXTS),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 3),
        "stdev_latency_ms": round(statistics.stdev(latencies_ms), 3)
        if len(latencies_ms) > 1
        else 0.0,
        "samples": rows,
        "note": "Fixed in-repo strings; use for smoke/regression only, not paper Table 2 numbers.",
    }
    print(json.dumps(summary, indent=2))

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {outp.resolve()}")


if __name__ == "__main__":
    main()
