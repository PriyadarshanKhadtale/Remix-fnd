# Setup Guide

## Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend)
- Git

## Quick Start

### 1. Clone & Setup

```bash
cd REMIX_FND_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Prepare Data

Copy processed data from original project:
```bash
mkdir -p data/processed/fakenewsnet
cp ../REMIX_FND/data/processed/fakenewsnet/*.json data/processed/fakenewsnet/
```

### 3. Copy Trained Model

```bash
mkdir -p models/text_classifier
cp ../REMIX_FND/checkpoints/best_simple.pt models/text_classifier/best_model.pt
```

### 4. Start Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for API documentation.

### 5. Start Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

## Docker Setup

```bash
docker-compose up --build
```

## Configuration

Create `.env` file in project root:

```env
# App Settings
DEBUG=false
DEVICE=cpu

# Feature Toggles
ENABLE_TEXT_ANALYSIS=true
ENABLE_IMAGE_ANALYSIS=false
ENABLE_EVIDENCE_RETRIEVAL=false
ENABLE_AI_DETECTION=false
ENABLE_EXPLAINABILITY=true
```

## Training New Model

```bash
cd training/scripts
python train_text_model.py \
  --data_dir ../../data/processed/fakenewsnet \
  --output_dir ../../models/text_classifier \
  --epochs 5
```

## Troubleshooting

### Model not found
Ensure model exists at `models/text_classifier/best_model.pt`

### CUDA/MPS errors
Set `DEVICE=cpu` in `.env` or pass `--device cpu` to training script

### Import errors
Ensure you're in the virtual environment and all dependencies are installed

