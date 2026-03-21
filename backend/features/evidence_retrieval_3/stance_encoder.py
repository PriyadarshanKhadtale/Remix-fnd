"""
Cross-encoder stance model (SUPPORTS / REFUTES / NEUTRAL) for claim--evidence pairs.
Loads checkpoint from train_stance_cross_encoder.py when present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_STANCE_SCORER: Optional[Any] = None  # NeuralStanceScorer or False after failed load


def resolve_stance_checkpoint() -> Optional[Path]:
    env = os.environ.get("REMIX_STANCE_MODEL")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent.parent
    candidates.append(project_root / "models" / "stance_cross_encoder" / "best_model.pt")
    candidates.append(Path("/app/models/stance_cross_encoder/best_model.pt"))
    for p in candidates:
        if p.exists():
            return p
    return None


class NeuralStanceScorer:
    """DistilRoBERTa sequence-pair classifier → supports / refutes / neutral."""

    ID2LABEL = {0: "supports", 1: "refutes", 2: "neutral"}

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_name = ckpt.get("model_name", "distilroberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=int(ckpt.get("num_labels", 3)),
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, claim: str, evidence_passage: str) -> Tuple[str, Dict[str, Any]]:
        import torch

        claim = (claim or "")[:512]
        passage = (evidence_passage or "")[:512]
        enc = self.tokenizer(
            claim,
            passage,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())
        label = self.ID2LABEL.get(pred, "neutral")
        return label, {
            "supports": float(probs[0].item()),
            "refutes": float(probs[1].item()),
            "neutral": float(probs[2].item()) if probs.size(0) > 2 else 0.0,
        }


def get_stance_scorer(device: str = "cpu") -> Optional[NeuralStanceScorer]:
    global _STANCE_SCORER
    if _STANCE_SCORER is False:
        return None
    if _STANCE_SCORER is not None:
        return _STANCE_SCORER  # type: ignore[return-value]
    path = resolve_stance_checkpoint()
    if path is None:
        return None
    try:
        _STANCE_SCORER = NeuralStanceScorer(path, device=device)
    except Exception as e:
        print(f"⚠️ Stance model not loaded ({e}); using heuristic stance.")
        _STANCE_SCORER = False
        return None
    return _STANCE_SCORER
