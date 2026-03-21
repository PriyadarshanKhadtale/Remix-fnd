"""
Monte Carlo dropout uncertainty and evidence routing (REMIX-FND §2.1, Table 1).
Reference prototype: optional T>0 forward passes with dropout active at inference.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _set_dropout_train(module: nn.Module, train: bool) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train(train)


def predict_with_mc_dropout(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    T: int = 30,
    fake_class_index: int = 1,
) -> Dict[str, Any]:
    """
    Run T stochastic forward passes (dropout on). Returns mean/variance of class probs.

    Paper: T=30; use T=0 caller to mean single deterministic eval (no MC).
    """
    if T <= 0:
        model.eval()
        _set_dropout_train(model, False)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0]
        p_real = float(probs[0].item())
        p_fake = float(probs[fake_class_index].item())
        pred = int(torch.argmax(probs).item())
        return {
            "mean_real": p_real,
            "mean_fake": p_fake,
            "var_real": 0.0,
            "var_fake": 0.0,
            "var_decisive": 0.0,
            "pred": pred,
            "mc_samples": 1,
        }

    model.eval()
    _set_dropout_train(model, True)
    reals, fakes, decisive = [], [], []
    with torch.no_grad():
        for _ in range(T):
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0]
            pr = float(probs[0].item())
            pf = float(probs[fake_class_index].item())
            reals.append(pr)
            fakes.append(pf)
            pred_local = 1 if pf >= pr else 0
            decisive.append(pf if pred_local == 1 else pr)

    _set_dropout_train(model, False)
    model.eval()

    mean_real = float(statistics.mean(reals))
    mean_fake = float(statistics.mean(fakes))
    var_real = float(statistics.variance(reals)) if len(reals) > 1 else 0.0
    var_fake = float(statistics.variance(fakes)) if len(fakes) > 1 else 0.0
    var_decisive = float(statistics.variance(decisive)) if len(decisive) > 1 else 0.0
    pred = 1 if mean_fake >= mean_real else 0

    return {
        "mean_real": mean_real,
        "mean_fake": mean_fake,
        "var_real": var_real,
        "var_fake": var_fake,
        "var_decisive": var_decisive,
        "pred": pred,
        "mc_samples": T,
    }


def table1_depth_from_fake_variance(
    var_fake: float,
    bernoulli_scale: float = 0.25,
) -> int:
    """
    Map predictive variance on fake probability to retrieval depth {5, 10, 20}.
    Normalizes by ~max variance of Bernoulli (0.25) as a monotone calibration proxy (§2.1 Table 1).
    """
    if bernoulli_scale <= 0:
        bernoulli_scale = 0.25
    sigma_n = min(max(var_fake / bernoulli_scale, 0.0), 1.0)
    if sigma_n > 0.8:
        return 20
    if sigma_n > 0.5:
        return 10
    return 5


def evidence_fast_path(
    mean_decisive_prob: float,
    var_decisive: float,
    conf_threshold: float = 0.8,
    var_threshold: float = 0.02,
) -> bool:
    """
    Fast path: skip heavy evidence when the decisive class probability is high and stable (§2.1).
    mean_decisive_prob: max(mean_real, mean_fake) or probability of predicted class.
    """
    return mean_decisive_prob >= conf_threshold and var_decisive <= var_threshold


def confidence_from_means(mean_real: float, mean_fake: float, pred: int) -> float:
    """UI confidence 0-100 from MC means."""
    p = mean_fake if pred == 1 else mean_real
    return float(max(0.0, min(100.0, p * 100.0)))
