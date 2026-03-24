# REMIX-FND Backend

## Overview

FastAPI backend for REMIX-FND. There are **two** HTTP surfaces:

| Command | Use |
|---------|-----|
| `python run.py` or `uvicorn run:app` | **Full** pipeline: `/detect` with MC dropout option, multimodal fusion, DSRG, image, early exit, `/ai-detect`, `/evidence`. **Docker default** (full stack). |
| `uvicorn app.main:app` | **Modular** routers under `app/routes/` with `ENABLE_*` flags in [`app/config.py`](app/config.py). |

See repository root **[SCOPE.md](../SCOPE.md)** for stage order, evidence depth policy, and KB sources (LIAR primary; FEVER optional).

## Structure

```
backend/
├── app/                    # Modular FastAPI (app.main)
│   ├── main.py
│   ├── config.py
│   └── routes/
├── features/               # Feature modules
│   ├── text_analysis_1/
│   ├── image_analysis_2/
│   ├── evidence_retrieval_3/
│   ├── ai_detection_4/
│   ├── explainability_5/
│   ├── multimodal_fusion/
│   ├── routing/
│   └── early_exit/
├── run.py                  # Full-stack API (canonical for paper-aligned /detect)
├── run_lite.py             # Low-memory API (Render / REMIX_RENDER_LITE)
├── core/
├── requirements.txt
└── requirements-docker.txt
```

## Quick Start

```bash
pip install -r requirements.txt

# Full stack (recommended for paper-aligned behavior)
python run.py
# open http://localhost:8000/docs

# Or modular app only
uvicorn app.main:app --reload --port 8000
```

## API Endpoints (`run.py`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Pipeline status and module flags |
| POST | `/detect` | Orchestrated detection |
| POST | `/explain` | 3-tier explanation |
| POST | `/evidence` | Evidence retrieval |
| POST | `/ai-detect` | AI-generated text ensemble |
| POST | `/image-analyze` | Image analysis |

## Configuration

Environment variables (see **SCOPE.md** for full list):

```env
REMIX_VERACITY_CKPT=          # optional explicit path to veracity .pt
REMIX_VERACITY_RUN_ID=        # e.g. 20260321T191615Z — looks under benchmarks/runs/<id>/
REMIX_VERACITY_VARIANT=dann   # dann | baseline | diml → <variant>_veracity/best_model.pt
REMIX_MC_FAST_CONF=0.8        # MC evidence fast path: min decisive mean prob (§2.3)
REMIX_MC_FAST_VAR=0.02        # MC evidence fast path: max raw variance across passes
```

**Checkpoints vs benchmarks:** Indexed runs live under **`benchmarks/runs/`** with **`benchmarks/index.json`**. For the manuscript’s best logged in-domain veracity F1 on the FakeNewsNet-format split, use the **+DANN** artifact associated with run id **`20260321T191615Z`** as `REMIX_VERACITY_CKPT` when you have that `.pt` locally. DIML export: **`20260322T135657Z_diml`**. Domain-tag diagnostics in the paper used a **different** baseline checkpoint—see docstring in **`training/scripts/evaluate_ood_domains.py`**.

Modular app (`.env` for `app.main`):

```env
ENABLE_TEXT_ANALYSIS=true
ENABLE_EVIDENCE_RETRIEVAL=false
ENABLE_AI_DETECTION=false
ENABLE_EXPLAINABILITY=true
ENABLE_IMAGE_ANALYSIS=false
DEVICE=cpu
```
