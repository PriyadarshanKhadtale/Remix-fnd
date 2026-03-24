# 🔍 REMIX-FND v3.0

**Real-time Fake News Detection with Multi-Modal Analysis, Evidence Retrieval & Explainability**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Scope and documentation](#scope-and-documentation)
- [Features](#-features)
- [Datasets Used](#-datasets-used)
- [Techniques & Methods](#-techniques--methods)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Performance](#-performance)

---

## 🎯 Overview

REMIX-FND is a comprehensive fake news detection system implementing research paper methodologies:

- **Module 1 (MSCIM)**: Multi-Modal Social Context Intelligence
- **Module 2 (EVRS)**: Evidence-Based Verification & Retrieval System  
- **Module 3 (ELDS)**: Explainable LLM Detection & Defense

**Implementation:** Full reference stack is in **`backend/run.py`** (use this for manuscript-aligned `/detect`). The modular **`app.main`** app is a slimmer alternative; see below.

---

## Scope and documentation

- **[SCOPE.md](SCOPE.md)** — canonical API entry points (`run.py` vs `app.main` vs `run_lite`), `/detect` stage order, **evidence depth** (default softmax-linear *k* vs **MC dropout** Table 1 when `mc_dropout_passes > 0`), **LIAR vs FEVER** KB, and route-to-module map (manuscript Table A / §2.6 alignment).
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — three-stage pipeline, module paths, env vars, benchmark run ids.

### Paper-aligned runtime and checkpoints

| Goal | What to run | Notes |
|------|-------------|--------|
| Full orchestration (MC option, DSRG, multimodal, early exit) | From `backend/`: `python run.py` or `uvicorn run:app` | Same as Docker full stack. |
| Slim API only | `uvicorn app.main:app` | Feature flags in `backend/app/config.py`; not the full paper pipeline. |
| Best **logged** in-domain veracity F1 on manuscript split (n = 3253) | **`REMIX_VERACITY_CKPT`** = path to **+DANN** `.pt`, **or** **`REMIX_VERACITY_RUN_ID=20260321T191615Z`** + **`REMIX_VERACITY_VARIANT=dann`** after copying `dann_veracity/best_model.pt` under `benchmarks/runs/20260321T191615Z/` | See [`benchmarks/index.json`](benchmarks/index.json); JSON is in-repo; `.pt` is often local/Colab only. |
| Table 1–style evidence depth | `POST /detect` with **`check_evidence`: true** and **`mc_dropout_passes`: 30** | Default `mc_dropout_passes: 0` uses softmax *k* (5–20). Fast path: **`REMIX_MC_FAST_CONF`**, **`REMIX_MC_FAST_VAR`**. |
| DIML checkpoint | Export **`20260322T135657Z_diml`** + `train_diml.py` | Manuscript: archived short schedule **below** +DANN F1 on that split; tune epochs/λ if you need DIML to catch up. |
| Domain subset diagnostics | `training/scripts/evaluate_ood_domains.py` | Use the **same** checkpoint as the benchmark row you cite; Table 5 subset used **baseline** from **`20260322T135702Z_dann`**, not +DANN headline weights. |

---

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Text Classification** | ✅ Complete | DistilRoBERTa-based fake news detection (85.2% accuracy) |
| **Evidence Retrieval (RAG)** | ✅ Complete | FAISS + 12.8K fact-checked claims from LIAR dataset |
| **AI Content Detection** | ✅ Complete | 6-detector ensemble (perplexity, burstiness, linguistic patterns, repetition, vocabulary, HC3 retrieval similarity) |
| **Image Analysis** | ✅ Complete | Manipulation detection, metadata analysis |
| **3-Tier Explanations** | ✅ Complete | Novice, Intermediate, Expert levels |
| **Sentence Attribution** | ✅ Complete | Per-sentence contribution scores |
| **Early Exit** | ✅ Complete | Confidence-based routing for efficiency |
| **Docker Deployment** | ✅ Complete | Production-ready containerization |

---

## 📊 Datasets Used

### 1. FakeNewsNet (Fake News Detection - Training)

| Property | Value |
|----------|-------|
| **Purpose** | Train text classification model |
| **Size** | 21,000+ articles |
| **Sources** | PolitiFact, GossipCop |
| **Labels** | Real, Fake |
| **Used For** | DistilRoBERTa fine-tuning |

```
data/fakenewsnet/
├── politifact_real.csv
├── politifact_fake.csv
├── gossipcop_real.csv
└── gossipcop_fake.csv
```

### 2. LIAR Dataset (Evidence Retrieval - Knowledge Base)

| Property | Value |
|----------|-------|
| **Purpose** | Fact-checking knowledge base for RAG |
| **Size** | 12,836 claims |
| **Source** | PolitiFact fact-checks |
| **Labels** | true, mostly-true, half-true, barely-true, false, pants-fire |
| **Used For** | Evidence retrieval, stance classification |

```
data/fact_checking/
├── train.tsv (10,269 claims)
├── valid.tsv (1,284 claims)
└── test.tsv (1,283 claims)
```

**Label Distribution:**
- half-true: 2,123
- false: 1,998
- mostly-true: 1,966
- true: 1,683
- barely-true: 1,657
- pants-fire: 842

### 3. HC3 - Human ChatGPT Comparison (AI Detection - Reference)

| Property | Value |
|----------|-------|
| **Purpose** | Reference for AI-generated text patterns |
| **Size** | 24,322 QA pairs |
| **Content** | Human answers + ChatGPT answers |
| **Source** | Hello-SimpleAI/HC3 (Hugging Face) |
| **Used For** | AI detection feature analysis |

```
data/ai_detection/
└── hc3_sample.json (24,322 entries)
```

### Dataset Summary

| Dataset | Size | Domain | Purpose |
|---------|------|--------|---------|
| **FakeNewsNet** | 21K | News articles | Model training |
| **LIAR** | 12.8K | Political claims | Evidence KB |
| **HC3** | 24K | QA pairs | AI detection |
| **Hand-crafted** | 30 | Health, Science | Supplementary facts |

---

## 🔬 Techniques & Methods

### Module 1: Text Classification (MSCIM)

| Technique | Description |
|-----------|-------------|
| **Model** | DistilRoBERTa (distilled RoBERTa) |
| **Architecture** | Transformer encoder + classification head |
| **Training** | Fine-tuned on FakeNewsNet |
| **Input** | News text (max 128 tokens) |
| **Output** | Binary classification (Real/Fake) + confidence |

```python
# Model Architecture
DistilRoBERTa Encoder (768-dim)
    → Linear(768, 256) + ReLU + Dropout(0.1)
    → Linear(256, 2)
    → Softmax
```

### Module 2: Evidence Retrieval (EVRS)

| Technique | Description |
|-----------|-------------|
| **Search** | Hybrid (FAISS semantic + keyword) |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Index** | FAISS IndexFlatIP (cosine similarity) |
| **Depth Control** | Uncertainty-based adaptive (5-20 docs) |
| **Stance Classification** | Supports / Refutes / Neutral |

```python
# Retrieval Pipeline
Query → Generate Embedding
      → FAISS Search (semantic)
      → Keyword Search (fallback)
      → Merge & Re-rank
      → Stance Classification
      → Verdict Generation
```

### Module 3: AI Detection (ELDS)

| Detector | Technique | What It Measures |
|----------|-----------|------------------|
| **Perplexity** | Word frequency analysis | Text predictability |
| **Burstiness** | Word clustering patterns | Human writing irregularity |
| **Linguistic** | Regex pattern matching | AI-specific phrases |
| **Repetition** | N-gram analysis | Phrase repetition |
| **Vocabulary** | TTR, hapax legomena | Vocabulary richness |
| **HC3 similarity** | Trigram Jaccard vs bundled reference strings | Corpus overlap signal |

```python
# Ensemble (AIContentDetector): weighted average in [0, 1]; binary AI/human at 0.55

Weights (manuscript Table 2; sum = 1.0):
- Perplexity: 0.22
- Burstiness: 0.18
- Linguistic patterns: 0.18
- Repetition: 0.14
- Vocabulary richness: 0.13
- HC3 corpus similarity: 0.15
```

### Module 4: Image Analysis

| Technique | Description |
|-----------|-------------|
| **Metadata Analysis** | EXIF inspection, software detection |
| **Quality Analysis** | Compression artifacts, bit-per-pixel |
| **ELA Approximation** | Error level analysis for manipulation |
| **Consistency Check** | Text-image topic matching |

### Module 5: Explainability

| Level | Target Audience | Content |
|-------|-----------------|---------|
| **Novice** | General public | Emojis, simple tips, actionable advice |
| **Intermediate** | Informed users | Statistics, key sentences, patterns |
| **Expert** | Researchers | Full attribution, confidence intervals, weights |

### Optimization: Early Exit

```python
# Confidence-based routing
if confidence >= 90%:
    → Exit early (skip remaining modules)
else:
    → Continue to next module

Stages: Text → AI Detection → Evidence → Image
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        REMIX-FND v3.0                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📰 Input Text/Image                                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Module 1: MSCIM                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │    Text      │  │    Image     │  │   Adaptive   │   │   │
│  │  │   Encoder    │  │   Analyzer   │  │    Fusion    │   │   │
│  │  │DistilRoBERTa │  │  ELA/Meta    │  │              │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼  (Early Exit if confidence > 90%)                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Module 2: EVRS                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │    FAISS     │  │    LIAR      │  │   Stance     │   │   │
│  │  │   Search     │  │  12.8K KB    │  │  Classifier  │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Module 3: ELDS                              │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │   │
│  │  │Perplex │ │Bursty  │ │Linguis │ │Repeti  │ │Vocab   │ │   │
│  │  │ -ity   │ │ -ness  │ │ -tic   │ │ -tion  │ │Richness│ │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ │   │
│  │              6-Detector Ensemble                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           3-Tier Hierarchical Explanation                │   │
│  │         Novice │ Intermediate │ Expert                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  📊 Final Verdict: REAL/FAKE + Confidence + Explanation        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and start
cd REMIX_FND_v2
docker-compose up -d backend

# Wait for startup (~2 minutes for model loading)
docker logs remix_fnd_v2-backend-1 -f

# Test
curl http://localhost:8000/health
```

### Option 2: Local Development

```bash
# Backend (full paper-aligned API — same as Docker full stack)
cd backend
pip install -r requirements.txt
python run.py
# Alternative: uvicorn run:app --host 0.0.0.0 --port 8000
# Modular API only (feature flags in .env): uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Frontend: http://localhost:5173 (Vite default)

### Hosted API + local UI (e.g. Render)

The [Dockerfile](./Dockerfile) defaults to **lite** mode on Render (`RENDER=true`) so small instances stay up. **Full** PyTorch API: set `REMIX_FULL_STACK=1` and use enough RAM (about 2GB+).

1. Deploy the repo on Render (Web Service, root `Dockerfile`, context `.`).
2. `cd frontend && npm install && npm run dev` — **`frontend/.env.development`** already points at `https://remix-fnd.onrender.com`, so health/detect hit Render (not `localhost:3000/api`). For a **local** backend on :8000, add **`frontend/.env.development.local`** with `VITE_USE_LOCAL_API=1`.
3. **Fake News Detection** works on hosted lite; **AI** / **Fact Check** need the full backend locally or `REMIX_FULL_STACK` on Render.

### Option 3: Reproduce benchmarks (Google Colab T4)

1. Open [`colab/REMIX_FND_T4_Benchmarks.ipynb`](colab/REMIX_FND_T4_Benchmarks.ipynb) in [Google Colab](https://colab.research.google.com/) and enable **GPU** runtime.
2. Run all cells — **`run_all_benchmarks.py`** trains/evaluates **veracity (baseline + DANN)**, **stance on LIAR**, **AI ensemble micro-benchmark**, and **encoder latency**; writes everything under **`benchmark_runs/<timestamp>/`** plus **`manifest.json`** (cell output logs stdout; JSON files are for download).
3. Veracity-only shortcut: `python training/scripts/run_benchmarks.py --device auto` (see [`training/README.md`](training/README.md)).

---

## 🔌 API Reference

Full **`/detect`** orchestration (MC dropout, multimodal flags, DSRG, early exit) is served by **`run.py`** (`python run.py` or `uvicorn run:app`). The **`app.main`** app exposes the same route names with a slimmer pipeline; see [SCOPE.md](SCOPE.md).

### POST /detect

Full fake news detection with optional stages. By default `mc_dropout_passes` is **0** (evidence depth uses softmax-linear *k*); set **`mc_dropout_passes`** (e.g. 30) with **`check_evidence: true`** for Table 1–style depth and fast-path rules.

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news headline here",
    "include_explanation": true,
    "explanation_level": "intermediate",
    "check_ai_generated": true,
    "check_evidence": true,
    "enable_early_exit": true,
    "mc_dropout_passes": 0
  }'
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 87.5,
  "feature_scores": {
    "text_analysis": 87.5,
    "ai_detection": 25.0,
    "evidence_retrieval": 90.0
  },
  "ai_analysis": {
    "is_ai_generated": false,
    "verdict": "Likely human-written"
  },
  "evidence": {
    "verdict": "likely_false",
    "evidence": [...]
  },
  "explanation": {
    "level": "intermediate",
    "features_used": [...]
  }
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with module status |
| `/explain` | POST | 3-tier explanation only |
| `/ai-detect` | POST | AI detection only |
| `/evidence` | POST | Evidence retrieval only |
| `/image-analyze` | POST | Image analysis only |

---

## 📈 Performance

### Model metrics (manuscript Table 4 / B, FakeNewsNet-format test n = 3253)

| Setting | Accuracy | Weighted F1 | Macro-F1 |
|---------|----------|-------------|----------|
| Baseline (measured) | 85.74% | 85.22 | 0.793 |
| + DANN (measured) | 85.95% | **85.47** | 0.797 |
| DIML trainer 2 ep (measured) | 84.94% | 84.58 | 0.786 |

Exports indexed under [`benchmarks/index.json`](benchmarks/index.json). Your local `best_model.pt` may differ unless copied from a named export.

### System performance

| Metric | Value |
|--------|-------|
| **Full API path** | ~400–500 ms (manuscript §3.5) |
| **Early exit** | ~200 ms (high confidence) |
| **Knowledge base** | ~12.8K LIAR-derived + hand entries |
| **AI detectors** | 6 (ensemble micro ~4 ms mean on fixed strings, Table B) |

---

## 📁 Project Structure

```
REMIX_FND_v2/
├── backend/
│   ├── features/
│   │   ├── text_analysis_1/        # Text classification
│   │   ├── image_analysis_2/       # Image manipulation detection
│   │   ├── evidence_retrieval_3/   # FAISS + LIAR RAG
│   │   ├── ai_detection_4/         # 6-detector ensemble
│   │   ├── explainability_5/       # 3-tier explanations
│   │   ├── routing/                # MC dropout uncertainty (Table 1 depth)
│   │   ├── multimodal_fusion/      # Text + image + social + temporal
│   │   └── early_exit/             # Confidence routing
│   ├── data_fact_checking/         # LIAR dataset
│   ├── data_ai_detection/          # HC3 dataset
│   ├── run.py                      # Main API server
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── styles/
│   ├── package.json
│   └── Dockerfile
│
├── models/
│   └── text_classifier/
│       └── best_model.pt           # Trained model (85.2% acc)
│
├── data/
│   ├── fact_checking/              # LIAR dataset
│   └── ai_detection/               # HC3 dataset
│
├── training/scripts/               # train_*, evaluate_*, benchmarks
├── benchmarks/runs/                # Frozen JSON exports (+ local .pt)
├── benchmarks/index.json
├── docker-compose.yml
├── SCOPE.md
└── README.md
```

---

## 🔮 Future work (extensions beyond frozen benchmarks)

The reference stack already includes DANN/DIML trainers, MC routing, DSRG, and the six-detector ensemble. Directions called out in the manuscript (§4.2) include **multilingual** coverage, **bias-aware** stratified evaluation, **cheaper uncertainty proxies** than full MC, **longitudinal** revalidation, and **quantization** for edge deployment—not all are implemented as automated benchmarks (`manifest.json` notes e.g. paraphrase-attack suites as not scripted).

---

## 📚 References

- **FakeNewsNet**: Shu et al., "FakeNewsNet: A Data Repository for Fake News Research"
- **LIAR**: Wang, "Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection"
- **HC3**: Guo et al., "How Close is ChatGPT to Human Experts?"
- **DistilRoBERTa**: Sanh et al., "DistilBERT, a distilled version of BERT"

---

## 📄 License

MIT License - See LICENSE file

---

Built with ❤️ for fighting misinformation
