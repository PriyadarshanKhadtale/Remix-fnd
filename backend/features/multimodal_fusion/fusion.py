"""
Attention-style fusion (Eqs. 4–5 spirit): project each modality to a common dim,
score with a learned query, softmax over available modalities, convex mix of
fake-likelihood hints, then optional small MLP. Text branch dominates by default init.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def _parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s or not str(s).strip():
        return None
    try:
        raw = str(s).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class MultimodalFusion(nn.Module):
    """4-modality gated fusion → adjusted fake probability in [0, 1]."""

    def __init__(self, text_dim: int = 768, hidden: int = 64):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden),
            nn.Tanh(),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
        )
        self.soc_proj = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
        )
        self.tmp_proj = nn.Sequential(
            nn.Linear(3, hidden),
            nn.Tanh(),
        )
        self.query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # Prior logits for [text, image, social, temporal] — text strongest
        self.modality_logits = nn.Parameter(torch.tensor([2.5, 0.8, 0.5, 0.5]))
        self.mixer = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        text_cls: torch.Tensor,
        image_vec: torch.Tensor,
        social_vec: torch.Tensor,
        temporal_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Training / introspection: attention-pooled representation → logit."""
        t = self.text_proj(text_cls)
        i = self.img_proj(image_vec)
        s = self.soc_proj(social_vec)
        tm = self.tmp_proj(temporal_vec)
        stack = torch.stack([t, i, s, tm], dim=1)
        q = self.query.view(1, 1, -1)
        scores = (stack * q).sum(dim=-1) / self.temperature.clamp(min=0.2)
        scores = scores + self.modality_logits.view(1, -1)
        scores = scores.masked_fill(mask < 0.5, -1e4)
        attn = torch.softmax(scores, dim=-1)
        fused = (attn.unsqueeze(-1) * stack).sum(dim=1)
        return self.mixer(fused).squeeze(-1)


def modality_fake_hints(
    _text_fake_prob: float,
    manipulation_score: Optional[float],
    consistency_score: Optional[float],
    social: Optional[Dict[str, Any]],
    published_at_iso: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build batch-1 vectors. Returns (text_cls_dummy, img_3, soc_4, tmp_3, mask[4]).
    text_cls is filled by caller; here we pass zeros for text dim — run.py overwrites.
    """
    # Image: manipulation pushes fake; consistency with text pulls real (invert)
    m = (manipulation_score or 0.0) / 100.0
    q = 0.5
    c = (consistency_score or 50.0) / 100.0
    img = torch.tensor([[m, q, 1.0 - c]], dtype=torch.float32)

    # Social defaults
    likes = float(social.get("likes", 0.0) or 0.0) if social else 0.0
    shares = float(social.get("shares", 0.0) or 0.0) if social else 0.0
    comments = float(social.get("comments", 0.0) or 0.0) if social else 0.0
    verified = 1.0 if (social and social.get("account_verified")) else 0.0
    # log1p scale viral counts (rough)
    soc = torch.tensor(
        [[math.log1p(likes), math.log1p(shares), math.log1p(comments), verified]],
        dtype=torch.float32,
    )

    # Temporal: hours since publish, sin/cos weekday
    now = datetime.now(timezone.utc)
    dt = _parse_iso_dt(published_at_iso)
    if dt is None:
        tmp = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        m_tmp = 0.0
    else:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = max(0.0, (now - dt).total_seconds() / 3600.0)
        h_norm = min(1.0, hours / (24.0 * 365.0))
        wd = dt.weekday()
        tmp = torch.tensor(
            [[h_norm, math.sin(2 * math.pi * wd / 7.0), math.cos(2 * math.pi * wd / 7.0)]],
            dtype=torch.float32,
        )
        m_tmp = 1.0

    m_img = 1.0 if manipulation_score is not None else 0.0
    m_soc = 1.0 if social and any(social.get(k) for k in ("likes", "shares", "comments", "account_verified")) else 0.0
    mask = torch.tensor([[1.0, m_img, m_soc, m_tmp]], dtype=torch.float32)

    text_placeholder = torch.zeros(1, 768, dtype=torch.float32)
    return text_placeholder, img, soc, tmp, mask


def fuse_detection_signals(
    module: MultimodalFusion,
    text_cls: torch.Tensor,
    text_fake_prob: float,
    manipulation_score: Optional[float],
    consistency_score: Optional[float],
    social: Optional[Dict[str, Any]],
    published_at_iso: Optional[str],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """
    Blend text fake probability with other modalities using attention gates.
    Returns (fused_fake_prob_0_100, debug dict).
    """
    _, img, soc, tmp, mask = modality_fake_hints(
        text_fake_prob,
        manipulation_score,
        consistency_score,
        social,
        published_at_iso,
    )
    text_h = text_cls.to(device)
    img = img.to(device)
    soc = soc.to(device)
    tmp = tmp.to(device)
    mask = mask.to(device)

    b = text_h.size(0)
    t = module.text_proj(text_h)
    i = module.img_proj(img)
    s = module.soc_proj(soc)
    tm = module.tmp_proj(tmp)
    stack = torch.stack([t, i, s, tm], dim=1)
    q = module.query.view(1, 1, -1)
    scores = (stack * q).sum(dim=-1) / module.temperature.clamp(min=0.2)
    scores = scores + module.modality_logits.view(1, -1)
    scores = scores.masked_fill(mask < 0.5, -1e4)
    attn = torch.softmax(scores, dim=-1)

    p_text = text_fake_prob / 100.0
    p_img = (manipulation_score or 0.0) / 100.0
    viral = float((soc[0, 0] + soc[0, 1] + soc[0, 2]).item())
    verified = float(soc[0, 3].item())
    p_soc = 0.5 + 0.1 * min(1.0, viral / 10.0) - 0.15 * verified
    p_soc = max(0.0, min(1.0, p_soc))
    if mask[0, 2] < 0.5:
        p_soc = p_text
    hours_norm = float(tmp[0, 0].item())
    p_tmp = 0.5 + 0.15 * hours_norm
    if mask[0, 3] < 0.5:
        p_tmp = p_text

    mix = (
        attn[0, 0] * p_text
        + attn[0, 1] * p_img
        + attn[0, 2] * p_soc
        + attn[0, 3] * p_tmp
    )
    fused = float(mix.item()) * 100.0
    fused = max(0.0, min(100.0, fused))
    dbg = {
        "attention_text": float(attn[0, 0].item()),
        "attention_image": float(attn[0, 1].item()),
        "attention_social": float(attn[0, 2].item()),
        "attention_temporal": float(attn[0, 3].item()),
        "p_text_fake": p_text,
        "p_image_fake_hint": p_img,
    }
    return fused, dbg


_fusion: Optional[MultimodalFusion] = None


def get_multimodal_fusion(device: str = "cpu") -> MultimodalFusion:
    global _fusion
    if _fusion is None:
        m = MultimodalFusion()
        ckpt = os.environ.get("REMIX_MULTIMODAL_FUSION_CKPT")
        paths = []
        if ckpt:
            paths.append(Path(ckpt))
        here = Path(__file__).resolve().parent.parent.parent.parent
        paths.append(here / "models" / "multimodal_fusion" / "best_model.pt")
        paths.append(Path("/app/models/multimodal_fusion/best_model.pt"))
        for p in paths:
            if p.exists():
                blob = torch.load(p, map_location="cpu")
                sd = blob.get("model_state_dict", blob)
                m.load_state_dict(sd, strict=False)
                break
        m.to(device)
        m.eval()
        _fusion = m
    return _fusion
