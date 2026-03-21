"""
REMIX-FND Backend Runner - Lite Version
========================================
Lightweight version that loads models on-demand to avoid memory issues.
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir.parent)

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ============================================
# Configuration
# ============================================
APP_NAME = "REMIX-FND"
APP_VERSION = "3.0.0-lite"

# ============================================
# Request/Response Models
# ============================================
class DetectionRequest(BaseModel):
    text: str
    include_explanation: bool = False
    explanation_level: str = "intermediate"

class ExplainRequest(BaseModel):
    text: str
    level: str = "intermediate"

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title=APP_NAME,
    description="REMIX-FND: Real-time Fake News Detection System (Lite Mode)",
    version=APP_VERSION,
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    print(f"\n{'='*60}")
    print(f"🚀 {APP_NAME} v{APP_VERSION}")
    print(f"   Running in LITE mode (models load on-demand)")
    print(f"{'='*60}")
    print("✅ Server ready!")
    print(f"📚 API Docs: http://localhost:8000/docs\n")

@app.get("/")
def root():
    """API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "mode": "lite",
        "status": "running",
        "endpoints": {
            "POST /detect": "Fake news detection (rule-based)",
            "POST /explain": "Get explanation",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "mode": "lite",
        "features": {
            "text_analysis": True,
            "rule_based": True
        }
    }

@app.post("/detect")
def detect(request: DetectionRequest):
    """
    Lightweight fake news detection using rule-based approach.
    No heavy ML models to avoid memory issues.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    text_lower = request.text.lower()
    
    # Rule-based detection
    suspicious_words = [
        'shocking', 'exposed', 'secret', 'conspiracy', 'breaking', 
        'urgent', 'bombshell', 'miracle', 'cure', 'revealed',
        'they don\'t want you to know', 'doctors hate', 'scientists discover'
    ]
    
    clickbait_patterns = [
        'you won\'t believe', 'this will shock you', 'what happens next',
        'number', 'will blow your mind', 'the truth about'
    ]
    
    # Scoring
    suspicious_count = sum(1 for word in suspicious_words if word in text_lower)
    clickbait_count = sum(1 for pattern in clickbait_patterns if pattern in text_lower)
    
    # Check for excessive caps and exclamation
    caps_ratio = sum(1 for c in request.text if c.isupper()) / max(len(request.text), 1)
    exclamation_count = request.text.count('!')
    
    # Calculate score
    score = 0
    score += suspicious_count * 10
    score += clickbait_count * 15
    score += min(caps_ratio * 100, 20)
    score += min(exclamation_count * 5, 15)
    
    # Determine prediction
    fake_prob = min(score, 95)
    real_prob = 100 - fake_prob
    prediction = "FAKE" if fake_prob > 50 else "REAL"
    confidence = max(fake_prob, real_prob)
    
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "processing_time_ms": (time.time() - start_time) * 1000,
        "mode": "rule-based-lite",
        "indicators_found": {
            "suspicious_words": suspicious_count,
            "clickbait_patterns": clickbait_count,
            "excessive_caps": caps_ratio > 0.3,
            "excessive_exclamation": exclamation_count > 3
        }
    }
    
    if request.include_explanation:
        explanation_text = []
        if suspicious_count > 0:
            explanation_text.append(f"Found {suspicious_count} suspicious words commonly used in fake news")
        if clickbait_count > 0:
            explanation_text.append(f"Detected {clickbait_count} clickbait patterns")
        if caps_ratio > 0.3:
            explanation_text.append("Excessive use of capital letters detected")
        if exclamation_count > 3:
            explanation_text.append(f"Excessive use of exclamation marks ({exclamation_count})")
        
        if not explanation_text:
            explanation_text.append("No strong indicators of fake news detected")
        
        result["explanation"] = {
            "summary": " | ".join(explanation_text),
            "level": request.explanation_level,
            "confidence_reason": f"Based on {suspicious_count + clickbait_count} linguistic indicators"
        }
    
    return result

@app.post("/explain")
def explain(request: ExplainRequest):
    """Get explanation."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return {
        "explanation": "This is a rule-based analysis. For full ML-based explanation, use the full version.",
        "level": request.level,
        "mode": "lite"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
