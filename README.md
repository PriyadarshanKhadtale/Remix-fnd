# 🔍 REMIX-FND v3.0

**Real-time Fake News Detection with Multi-Modal Analysis, Evidence Retrieval & Explainability**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📋 Table of Contents

- [Overview](#-overview)
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

**Implementation Status: ~70% of paper architecture**

---

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Text Classification** | ✅ Complete | DistilRoBERTa-based fake news detection (85.2% accuracy) |
| **Evidence Retrieval (RAG)** | ✅ Complete | FAISS + 12.8K fact-checked claims from LIAR dataset |
| **AI Content Detection** | ✅ Complete | 5-detector ensemble (perplexity, burstiness, linguistic) |
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

```python
# Ensemble Voting
Final Score = Σ (detector_score × weight) / Σ weights

Weights:
- Perplexity: 0.25
- Burstiness: 0.20
- Linguistic: 0.20
- Repetition: 0.15
- Vocabulary: 0.20
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
│  │              5-Detector Ensemble                         │   │
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
# Backend
cd backend
pip install -r requirements.txt
python run.py

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
2. In `frontend`, copy `frontend/.env.example` to `.env.local` and set:
   `VITE_API_BASE=https://remix-fnd.onrender.com` (hosted **lite** API; no trailing slash)
3. `cd frontend && npm install && npm run dev` — open the Vite URL; **Fake News Detection** works on lite; **AI** / **Fact Check** tabs need the full backend locally or `REMIX_FULL_STACK` on Render.

---

## 🔌 API Reference

### POST /detect

Full fake news detection with all features.

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news headline here",
    "include_explanation": true,
    "explanation_level": "intermediate",
    "check_ai_generated": true,
    "check_evidence": true,
    "enable_early_exit": true
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

### Model Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 85.2% |
| **F1-Score** | ~80% |
| **Dataset** | FakeNewsNet (21K) |
| **Model** | DistilRoBERTa |

### System Performance

| Metric | Value |
|--------|-------|
| **Latency** | 400-500ms (full pipeline) |
| **Early Exit** | ~200ms (high confidence) |
| **Knowledge Base** | 12,819 facts |
| **AI Detectors** | 5 ensemble |

---

## 📁 Project Structure

```
REMIX_FND_v2/
├── backend/
│   ├── features/
│   │   ├── text_analysis_1/        # Text classification
│   │   ├── image_analysis_2/       # Image manipulation detection
│   │   ├── evidence_retrieval_3/   # FAISS + LIAR RAG
│   │   ├── ai_detection_4/         # 5-detector ensemble
│   │   ├── explainability_5/       # 3-tier explanations
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
├── docker-compose.yml
└── README.md
```

---

## 🔮 Future Work (Remaining ~30%)

| Feature | Requirement |
|---------|-------------|
| Social Context | Twitter/X API access |
| Temporal Graphs | Real propagation data |
| Domain-Invariant Meta-Learning | GPU training time |
| Adversarial Training | Attack data generation |
| Model Quantization | INT8 optimization |

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
