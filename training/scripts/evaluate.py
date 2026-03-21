"""
Evaluate a saved veracity checkpoint on test.json (FakeNewsNet-style).
Supports baseline TextClassifier and domain-adversarial checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from features.text_analysis_1.domain_adversarial import DomainAdversarialClassifier  # noqa: E402

# Reuse training dataset layout
sys.path.insert(0, str(Path(__file__).parent))
from train_text_model import NewsDataset, TextClassifier  # noqa: E402


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    if ckpt.get("model_type") == "domain_adversarial" or any(k.startswith("veracity.") for k in sd):
        nd = int(ckpt.get("num_domains", 2))
        m = DomainAdversarialClassifier(num_domains=nd)
        m.load_state_dict(sd, strict=True)
    else:
        m = TextClassifier()
        m.load_state_dict(sd, strict=False)
    m.to(device)
    m.eval()
    return m


@torch.no_grad()
def run_eval(model, loader, device):
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="eval"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(batch["label"].numpy().tolist())
    return y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--test_data", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate on first N test examples only (quick demo)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model_path = Path(args.model_path)
    test_path = Path(args.test_data)
    if not model_path.exists() or not test_path.exists():
        raise SystemExit("model_path or test_data missing")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    ds = NewsDataset(test_path, tokenizer, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = load_model(model_path, device)
    y_true, y_pred = run_eval(model, loader, device)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Macro F1: {100 * f1_score(y_true, y_pred, average='macro'):.2f}%")


if __name__ == "__main__":
    main()
