"""
Evaluate stance cross-encoder on LIAR TSV (held-out split: test.tsv by default).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(SCRIPTS))
import core.torch_env  # noqa: E402, F401

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from device_util import device_pretty, resolve_device  # noqa: E402
from train_stance_cross_encoder import LiarStanceDataset, find_liar_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument(
        "--liar_split",
        type=str,
        default="test",
        choices=("train", "valid", "test"),
        help="Which LIAR TSV to score",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output_json", type=str, default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    liar_dir = find_liar_dir()
    tsv = liar_dir / f"{args.liar_split}.tsv"
    if not tsv.exists():
        raise SystemExit(f"Missing {tsv}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "distilroberta-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = LiarStanceDataset([tsv], tokenizer, max_samples=args.max_samples)
    if len(ds) < 10:
        raise SystemExit(f"Too few stance eval samples: {len(ds)}")

    loader = DataLoader(ds, batch_size=args.batch_size)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=int(ckpt.get("num_labels", 3))
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="stance_eval"):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(batch["labels"].numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nDevice: {device_pretty(device)}")
    print(f"Stance eval on {tsv.name}: n={len(y_true)}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Accuracy: {100 * acc:.2f}%  Macro-F1: {macro_f1:.4f}")

    if args.output_json:
        report = classification_report(
            y_true, y_pred, digits=6, output_dict=True, zero_division=0
        )
        out = {
            "task": "stance_liar",
            "liar_split": args.liar_split,
            "n": len(y_true),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "device": device_pretty(device),
            "checkpoint": str(ckpt_path.resolve()),
            "sklearn_report": report,
        }
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {outp.resolve()}")


if __name__ == "__main__":
    main()
