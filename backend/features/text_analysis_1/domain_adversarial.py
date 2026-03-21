"""
Domain-adversarial veracity classifier (DANN-style; paper DIML without inner-loop MAML).
Inference: returns only fake/real logits; domain head unused at deploy time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReversalFunction.apply(x, alpha)


class DomainAdversarialClassifier(nn.Module):
    def __init__(
        self,
        num_domains: int,
        model_name: str = "distilroberta-base",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.veracity = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )
        self.domain = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_domains),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        grl_alpha: float = 0.1,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:, 0, :]
        logits_v = self.veracity(h)
        if self.training:
            alpha = grl_alpha if grl_alpha > 0 else 1e-3
            logits_d = self.domain(grad_reverse(h, alpha))
            return logits_v, logits_d
        return logits_v
