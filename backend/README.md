# REMIX-FND Backend

## Overview

The backend API for the REMIX-FND fake news detection system. Built with FastAPI.

## Structure

```
backend/
├── app/                    # Main application
│   ├── main.py            # FastAPI entry point
│   ├── config.py          # Configuration
│   └── routes/            # API endpoints
│
├── features/              # Feature modules (self-contained)
│   ├── text_analysis_1/   # Text-based detection
│   ├── image_analysis_2/  # Image analysis (planned)
│   ├── evidence_retrieval_3/  # RAG fact-checking (planned)
│   ├── ai_detection_4/    # AI content detection (planned)
│   └── explainability_5/  # Explanation generation
│
├── core/                  # Shared utilities
├── requirements.txt       # Python dependencies
└── Dockerfile            # Container config
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload --port 8000

# Visit docs
open http://localhost:8000/docs
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/detect` | Detect fake news |
| POST | `/explain` | Get explanation |
| POST | `/evidence` | Retrieve evidence |

## Example Usage

```bash
# Detect fake news
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news headline here"}'
```

## Configuration

Set environment variables or create `.env` file:

```env
DEBUG=false
ENABLE_TEXT_ANALYSIS=true
ENABLE_EXPLAINABILITY=true
DEVICE=cpu
```

