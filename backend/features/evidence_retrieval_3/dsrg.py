"""
DSRG — Dynamic Source Reliability Graph (lightweight inference-time variant).

Builds an undirected graph from KB facts: sources that co-appear on the same fact
are linked. Reliability is a few steps of graph diffusion from institution priors,
then used to re-weight retrieval relevance scores (no external torch_geometric dep).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Substrings → prior reliability in [0, 1]
_HIGH_TRUST = (
    "who",
    "cdc",
    "fda",
    "nasa",
    "noaa",
    "ipcc",
    "nih",
    "cisa",
    "nist",
    "politifact",
    "snopes",
    "factcheck",
    "ap fact",
    "reuters",
    "bbc",
    "nature",
    "science",
    "lancet",
    "harvard",
    "mit ",
    "ieee",
    "esa",
    "university",
    "commission",
    "academy of sciences",
)
_LOW_TRUST = ("blog", "rumor", "anonymous", "telegram", "whatsapp", "unknown")


def _normalize_source_label(raw: str) -> str:
    s = (raw or "Unknown").strip()
    if not s:
        return "Unknown"
    return s[:120]


def _split_sources(source_field: str) -> List[str]:
    if not source_field:
        return ["Unknown"]
    parts = re.split(r"[,;&/]| and ", source_field, flags=re.IGNORECASE)
    out = [_normalize_source_label(p) for p in parts if p.strip()]
    return out if out else ["Unknown"]


def _prior_for_source(name: str) -> float:
    low = name.lower()
    if any(x in low for x in _LOW_TRUST):
        return 0.35
    if any(x in low for x in _HIGH_TRUST):
        return 0.92
    return 0.62


class SourceReliabilityGraph:
    """
    Symmetric adjacency + self-loops, diffusion smoothing of priors.
    """

    def __init__(self, facts: List[Dict[str, Any]], max_nodes: int = 400):
        self._source_to_idx: Dict[str, int] = {}
        self._idx_to_source: List[str] = []
        self._r: np.ndarray = np.array([], dtype=np.float64)
        self._S: Optional[np.ndarray] = None
        self._built = False
        if not facts:
            return
        source_fact_count: Dict[str, int] = defaultdict(int)
        for fact in facts:
            for s in set(_split_sources(str(fact.get("source", "Unknown")))):
                source_fact_count[s] += 1
        if not source_fact_count:
            return
        ranked = sorted(source_fact_count.keys(), key=lambda k: -source_fact_count[k])
        keep: Set[str] = set(ranked[:max_nodes])

        co_edges: Dict[Tuple[int, int], int] = defaultdict(int)
        for fact in facts:
            srcs = {s for s in set(_split_sources(str(fact.get("source", "Unknown")))) if s in keep}
            if not srcs:
                continue
            idxs = sorted(self._ensure_nodes(srcs))
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    if a > b:
                        a, b = b, a
                    co_edges[(a, b)] += 1

        n = len(self._idx_to_source)
        if n == 0:
            return

        adj = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(adj, 1.0)
        for (a, b), w in co_edges.items():
            adj[a, b] += float(w)
            adj[b, a] += float(w)

        deg = adj.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        self._S = adj / deg

        prior = np.array([_prior_for_source(self._idx_to_source[i]) for i in range(n)], dtype=np.float64)
        r = prior.copy()
        for _ in range(3):
            r = 0.55 * self._S @ r + 0.45 * prior
        self._r = np.clip(r, 0.08, 0.99)
        self._built = True

    def _ensure_nodes(self, names: Set[str]) -> List[int]:
        out = []
        for name in names:
            if name not in self._source_to_idx:
                self._source_to_idx[name] = len(self._idx_to_source)
                self._idx_to_source.append(name)
            out.append(self._source_to_idx[name])
        return out

    def reliability(self, source_name: str) -> float:
        if not self._built or self._r.size == 0:
            return _prior_for_source(source_name)
        key = _normalize_source_label(source_name)
        idx = self._source_to_idx.get(key)
        if idx is None:
            for sep in (",", ";", "&"):
                if sep in key:
                    parts = [_normalize_source_label(p) for p in key.split(sep)]
                    vals = [self._r[self._source_to_idx[p]] for p in parts if p in self._source_to_idx]
                    if vals:
                        return float(np.mean(vals))
            return _prior_for_source(key)
        return float(self._r[idx])

    def boost_score(self, relevance: float, source_name: str, mix: float = 0.5) -> float:
        rel = self.reliability(source_name)
        # Up-weight high-trust sources; damp low-trust
        w = mix * rel + (1.0 - mix) * 1.0
        return float(max(0.0, min(1.0, relevance * (0.55 + 0.45 * w))))


def build_dsrg_from_kb_facts(facts: List[Dict[str, Any]]) -> SourceReliabilityGraph:
    return SourceReliabilityGraph(facts)
