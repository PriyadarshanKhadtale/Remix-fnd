"""
Evaluate a saved veracity checkpoint on test.json (FakeNewsNet-style).
Supports baseline TextClassifier and domain-adversarial checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import core.torch_env  # noqa: E402, F401

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from features.text_analysis_1.domain_adversarial import DomainAdversarialClassifier  # noqa: E402

# Reuse training dataset layout
sys.path.insert(0, str(Path(__file__).parent))
from train_text_model import NewsDataset, TextClassifier  # noqa: E402


def load_model(model_path: Path, device: torch.device):
    # Always load tensors on CPU first; avoids MPS/cuda errors during load_state_dict.
    ckpt = torch.load(model_path, map_location="cpu")
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
    y_true, y_pred, _ = run_eval_with_scores(model, loader, device, collect_scores=False)
    return y_true, y_pred


@torch.no_grad()
def run_eval_with_scores(
    model,
    loader,
    device,
    collect_scores: bool = True,
) -> Tuple[List[int], List[int], Optional[List[float]]]:
    y_true, y_pred, y_score = [], [], []
    for batch in tqdm(loader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        pred = logits.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(batch["label"].numpy().tolist())
        if collect_scores:
            prob_fake = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
            y_score.extend(prob_fake)
    if not collect_scores:
        return y_true, y_pred, None
    return y_true, y_pred, y_score


def compute_metrics_bundle(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Scalar metrics for benchmarking / JSON export (binary veracity)."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    bundle: Dict[str, Any] = {
        "accuracy": acc,
        "accuracy_percent": round(100.0 * acc, 4),
        "macro_f1": macro_f1,
        "macro_f1_percent": round(100.0 * macro_f1, 4),
        "weighted_f1": weighted_f1,
        "precision_fake": float(p),
        "recall_fake": float(r),
        "f1_fake": float(f),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, output_dict=True, zero_division=0
        ),
    }
    if y_score is not None and len(set(y_true)) > 1:
        try:
            bundle["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            bundle["auroc"] = None
    else:
        bundle["auroc"] = None
    return bundle


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
    ap.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Write metrics JSON to this path (parent dirs created)",
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
    y_true, y_pred, y_score = run_eval_with_scores(model, loader, device, collect_scores=True)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"Macro F1: {100 * f1_score(y_true, y_pred, average='macro', zero_division=0):.2f}%")

    if args.json_out:
        out = {
            "model_path": str(model_path.resolve()),
            "test_data": str(test_path.resolve()),
            "n_examples": len(y_true),
            "device": str(device),
            "metrics": compute_metrics_bundle(y_true, y_pred, y_score),
        }
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote metrics JSON: {outp}")


if __name__ == "__main__":
    main()
