# Evidence Retrieval (EVRS)

## Overview

Retrieval-augmented evidence for claims: **semantic search** (Sentence-Transformers + FAISS when available) with **keyword fallback**, stance-aware scoring, optional **DSRG** reliability propagation, and uncertainty-based retrieval depth.

## Primary knowledge base

| Source | Role |
|--------|------|
| **LIAR** | Default — PolitiFact-scoped claims loaded into `ExpandedKnowledgeBase` in `retriever.py`. |
| **Hand-crafted facts** | Small curated set bundled with the retriever for health/science examples. |

## Optional / not default

| Source | Role |
|--------|------|
| **FEVER** | `load_fever_dataset.py` only — not used by `EvidenceRetriever()` by default. |

## Pipeline

```
Claim → embed or keyword match → FAISS / fallback → rank → stance + optional DSRG → verdict
```

See repository root **`SCOPE.md`** for how `/detect` chooses MC-based depth (Table 1) vs softmax-linear **k** when `mc_dropout_passes=0`.
