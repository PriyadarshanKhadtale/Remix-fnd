"""Resolve torch.device for training/eval (Colab T4, local MPS, CPU)."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    n = (name or "cpu").strip().lower()
    if n == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(n)


def device_pretty(dev: torch.device) -> str:
    if dev.type == "cuda" and torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return str(dev)
