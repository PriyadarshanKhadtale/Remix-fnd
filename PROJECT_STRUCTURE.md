# REMIX-FND Project Structure

```
REMIX_FND/
│
├── 📄 README.md                    # Project overview & quick start
├── 📄 docker-compose.yml           # Run entire stack with one command
├── 📄 .env.example                 # Environment variables template
│
│
├── 🎨 frontend/                    # FRONTEND - User Interface
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/            # Reusable UI components
│   │   │   ├── NewsInput.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── ExplanationPanel.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx           # Main detection page
│   │   │   └── About.jsx
│   │   ├── styles/
│   │   │   └── main.css
│   │   ├── utils/
│   │   │   └── api.js             # API calls to backend
│   │   ├── App.jsx
│   │   └── index.jsx
│   ├── package.json
│   └── README.md
│
│
├── ⚙️ backend/                     # BACKEND - All Server Logic
│   │
│   ├── app/                       # 🚀 Main Application Entry
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app initialization
│   │   ├── config.py             # All configuration settings
│   │   └── routes/               # API Endpoints
│   │       ├── __init__.py
│   │       ├── detect.py         # POST /detect - Main detection
│   │       ├── explain.py        # POST /explain - Get explanations
│   │       ├── evidence.py       # POST /evidence - Fact check
│   │       └── health.py         # GET /health - Health check
│   │
│   ├── features/                  # 🧩 FEATURE MODULES (Each Self-Contained)
│   │   │
│   │   ├── 1_text_analysis/      # 📝 Text-Based Detection
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # Neural network architecture
│   │   │   ├── predictor.py      # Prediction logic
│   │   │   ├── preprocessor.py   # Text cleaning & tokenization
│   │   │   └── README.md         # Feature documentation
│   │   │
│   │   ├── 2_image_analysis/     # 🖼️ Image-Based Detection
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # Image CNN architecture
│   │   │   ├── predictor.py      # Image prediction logic
│   │   │   ├── preprocessor.py   # Image transformations
│   │   │   └── README.md
│   │   │
│   │   ├── 3_evidence_retrieval/ # 🔍 RAG / Fact-Checking
│   │   │   ├── __init__.py
│   │   │   ├── retriever.py      # Search & retrieve evidence
│   │   │   ├── knowledge_base.py # Manage facts database
│   │   │   ├── embeddings.py     # Text embeddings for search
│   │   │   └── README.md
│   │   │
│   │   ├── 4_ai_detection/       # 🤖 AI-Generated Content Detection
│   │   │   ├── __init__.py
│   │   │   ├── detector.py       # Detect ChatGPT/AI text
│   │   │   ├── statistical.py    # Perplexity, burstiness metrics
│   │   │   └── README.md
│   │   │
│   │   └── 5_explainability/     # 💡 Explanation Generation
│   │       ├── __init__.py
│   │       ├── explainer.py      # Generate human explanations
│   │       ├── highlighter.py    # Highlight suspicious parts
│   │       ├── attention_viz.py  # Attention visualization
│   │       └── README.md
│   │
│   ├── core/                      # 🔧 Shared Utilities
│   │   ├── __init__.py
│   │   ├── base_model.py         # Base class for all models
│   │   ├── exceptions.py         # Custom exceptions
│   │   └── utils.py              # Helper functions
│   │
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile                # Backend container
│   └── README.md                 # Backend documentation
│
│
├── 🧠 models/                      # TRAINED MODEL FILES
│   ├── text_classifier/
│   │   └── best_model.pt         # Trained text model
│   ├── image_classifier/
│   │   └── (future)
│   └── ai_detector/
│       └── (future)
│
│
├── 📊 data/                        # ALL DATA FILES
│   ├── raw/                       # Original untouched datasets
│   │   └── fakenewsnet/
│   ├── processed/                 # Cleaned & ready-to-use data
│   │   └── fakenewsnet/
│   │       ├── train.json
│   │       ├── val.json
│   │       └── test.json
│   └── knowledge_base/            # Facts for evidence retrieval
│       └── facts.json
│
│
├── 🏋️ training/                    # MODEL TRAINING
│   ├── scripts/
│   │   ├── train_text_model.py
│   │   ├── train_image_model.py
│   │   └── evaluate.py
│   ├── notebooks/
│   │   └── experiments.ipynb
│   └── configs/
│       └── training_config.yaml
│
│
└── 📚 docs/                        # DOCUMENTATION
    ├── API_REFERENCE.md           # API endpoints documentation
    ├── SETUP_GUIDE.md             # Installation instructions
    ├── ARCHITECTURE.md            # System design explanation
    └── CONTRIBUTING.md            # How to contribute
```

## Folder Naming Convention

- **Numbers prefix (1_, 2_, etc.)**: Shows the order/priority of features
- **Lowercase with underscores**: Python-friendly naming
- **Self-explanatory names**: No abbreviations that need explanation

## Feature Independence

Each feature folder in `backend/features/` is **self-contained**:
- Has its own model, predictor, and preprocessor
- Has its own README explaining what it does
- Can be developed/tested independently
- Can be enabled/disabled without affecting others

