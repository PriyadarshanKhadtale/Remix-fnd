"""Routing and uncertainty helpers (paper §2.1 MC dropout, Table 1 depths)."""

from .mc_uncertainty import (
    predict_with_mc_dropout,
    table1_depth_from_fake_variance,
    evidence_fast_path,
)

__all__ = [
    "predict_with_mc_dropout",
    "table1_depth_from_fake_variance",
    "evidence_fast_path",
]
