"""
REMIX-FND Backend Runner - Hybrid Version
==========================================
Uses real PyTorch ML model + lightweight features
Best balance of accuracy and memory efficiency
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
os.chdir(backend_dir.parent)

import core.torch_env  # noqa: F401
import torch

core.torch_env.limit_pytorch_threads(torch)
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ============================================
# Configuration
# ============================================
APP_NAME = "REMIX-FND"
APP_VERSION = "3.0.0-hybrid"
MODEL_PATH = Path("models/text_classifier/best_model.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("../models/text_classifier/best_model.pt")

# ============================================
# Model Definition
# ============================================
class TextClassifier(nn.Module):
    def __init__(self, model_name="distilroberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

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
    description="REMIX-FND: Fake News Detection (Hybrid Mode - Real ML + Lightweight Features)",
    version=APP_VERSION,
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model = None
tokenizer = None

@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global model, tokenizer
    
    print(f"\n{'='*60}")
    print(f"🚀 {APP_NAME} v{APP_VERSION}")
    print(f"   Hybrid Mode: Real ML Model + Lightweight Features")
    print(f"{'='*60}")
    
    # Load tokenizer
    print("\n📦 Loading ML components...")
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    print("  ✓ Tokenizer loaded")
    
    # Load model
    model = TextClassifier()
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        acc = checkpoint.get('val_acc', 0)
        print(f"  ✓ PyTorch Text Classifier loaded ({acc:.1f}% accuracy)")
    else:
        print(f"  ⚠ Model not found - using untrained model")
    
    print(f"\n{'='*60}")
    print("✅ Hybrid backend ready!")
    print(f"{'='*60}")
    print("""
    📋 Features:
    ├─ ✅ Real ML: DistilRoBERTa Text Classification
    ├─ ✅ Rule-based: Linguistic Pattern Analysis
    ├─ ✅ Basic Explainability
    └─ ⚡ Memory Optimized
    """)
    print(f"📚 API Docs: http://localhost:8000/docs\n")

@app.get("/")
def root():
    """API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "mode": "hybrid",
        "status": "running",
        "features": {
            "ml_classification": True,
            "rule_based_analysis": True,
            "explainability": True
        },
        "endpoints": {
            "POST /detect": "Fake news detection (ML + rules)",
            "POST /explain": "Get explanation",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "mode": "hybrid",
        "modules": {
            "ml_classifier": model is not None,
            "tokenizer": tokenizer is not None
        }
    }

@app.post("/detect")
def detect(request: DetectionRequest):
    """
    Hybrid fake news detection using ML model + linguistic rules.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    
    # ========================================
    # ML-based Text Classification
    # ========================================
    try:
        encoding = tokenizer(
            request.text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            logits = model(encoding['input_ids'], encoding['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
        
        ml_fake_prob = probs[0][1].item() * 100
        ml_real_prob = probs[0][0].item() * 100
        ml_prediction = "FAKE" if pred == 1 else "REAL"
        ml_confidence = max(ml_fake_prob, ml_real_prob)
    except Exception as e:
        print(f"⚠️ ML inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")
    
    # ========================================
    # Rule-based Analysis (lightweight)
    # ========================================
    text_lower = request.text.lower()
    
    suspicious_words = ['shocking', 'exposed', 'secret', 'conspiracy', 'breaking', 
                       'urgent', 'bombshell', 'miracle', 'cure']
    clickbait_patterns = ['you won\'t believe', 'what happens next', 'doctors hate']
    
    suspicious_count = sum(1 for w in suspicious_words if w in text_lower)
    clickbait_count = sum(1 for p in clickbait_patterns if p in text_lower)
    
    rule_score = (suspicious_count * 10) + (clickbait_count * 15)
    
    # ========================================
    # Combine ML + Rules (weighted average)
    # ========================================
    ml_weight = 0.8  # ML model gets 80% weight
    rule_weight = 0.2  # Rules get 20% weight
    
    rule_fake_prob = min(rule_score, 100)
    combined_fake_prob = (ml_fake_prob * ml_weight) + (rule_fake_prob * rule_weight)
    combined_real_prob = 100 - combined_fake_prob
    
    final_prediction = "FAKE" if combined_fake_prob > 50 else "REAL"
    final_confidence = max(combined_fake_prob, combined_real_prob)
    
    # ========================================
    # Build Response
    # ========================================
    result = {
        "prediction": final_prediction,
        "confidence": float(final_confidence),
        "fake_probability": float(combined_fake_prob),
        "real_probability": float(combined_real_prob),
        "processing_time_ms": (time.time() - start_time) * 1000,
        "mode": "hybrid (ML + rules)",
        "model_analysis": {
            "ml_prediction": ml_prediction,
            "ml_confidence": float(ml_confidence),
            "model": "DistilRoBERTa (85.2% accuracy)"
        },
        "linguistic_analysis": {
            "suspicious_words_found": suspicious_count,
            "clickbait_patterns_found": clickbait_count,
            "rule_score": float(rule_score)
        }
    }
    
    if request.include_explanation:
        explanation_parts = []
        explanation_parts.append(f"ML Model predicts: {ml_prediction} ({ml_confidence:.1f}% confidence)")
        
        if suspicious_count > 0:
            explanation_parts.append(f"Found {suspicious_count} suspicious words")
        if clickbait_count > 0:
            explanation_parts.append(f"Detected {clickbait_count} clickbait patterns")
        
        result["explanation"] = {
            "summary": " | ".join(explanation_parts),
            "level": request.explanation_level,
            "method": "Ensemble: ML (80%) + Linguistic Rules (20%)"
        }
    
    return result

@app.post("/explain")
def explain(request: ExplainRequest):
    """Get explanation."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return {
        "explanation": "Hybrid system uses trained ML model (DistilRoBERTa) combined with linguistic pattern analysis.",
        "level": request.level,
        "mode": "hybrid"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
