"""
REMIX-FND Backend Runner v3
===========================
Complete API with all paper features:
- Fake News Detection (MSCIM)
- Evidence Retrieval with FAISS (EVRS)  
- AI Content Detection (ELDS)
- Image Analysis with manipulation detection
- Hierarchical 3-Tier Explanations
- Early Exit & Confidence Routing
- Sentence-Level Attribution
"""

import os
import sys
import time
import base64
import asyncio
import traceback
from pathlib import Path

# Setup paths (needed before core.torch_env is importable)
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))
# Local dev: models/ lives next to backend/. Docker/Render: both are under /app.
repo_root = backend_dir if (backend_dir / "models").exists() else backend_dir.parent
os.chdir(repo_root)

import core.torch_env  # noqa: F401 — sets OMP/MKL env before torch
import torch

core.torch_env.limit_pytorch_threads(torch)
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import feature modules
from features.explainability_5.explainer import HierarchicalExplainer
from features.ai_detection_4.detector import AIContentDetector
from features.evidence_retrieval_3.retriever import EvidenceRetriever, FAISS_AVAILABLE, EMBEDDINGS_AVAILABLE
from features.image_analysis_2.analyzer import ImageAnalyzer
from features.early_exit.router import EarlyExitRouter, AdaptivePipeline
from features.routing.mc_uncertainty import (
    predict_with_mc_dropout,
    table1_depth_from_fake_variance,
    evidence_fast_path,
    confidence_from_means,
)
from features.text_analysis_1.domain_adversarial import DomainAdversarialClassifier

def to_serializable(obj):
    """
    Convert numpy / odd scalars so FastAPI's jsonable_encoder can build JSON.
    (jsonable_encoder does not accept np.bool_/np.generic; causes 500 on /detect with ai_analysis.)
    """
    import math
    from enum import Enum

    import numpy as np
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, set):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # All numpy scalar types (bool_, int32, float64, numpy.bool on NumPy 2.x, etc.)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    return obj


# ============================================
# Configuration
# ============================================
APP_NAME = "REMIX-FND"
APP_VERSION = "3.0.0"
# Model path - works both in Docker and local
MODEL_PATH = Path("/app/models/text_classifier/best_model.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/text_classifier/best_model.pt")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("../models/text_classifier/best_model.pt")
if os.environ.get("REMIX_VERACITY_CKPT"):
    MODEL_PATH = Path(os.environ["REMIX_VERACITY_CKPT"])

# Paper thresholds
EARLY_EXIT_THRESHOLD = 0.90
EVIDENCE_DEPTH_MIN = 5
EVIDENCE_DEPTH_MAX = 20
# §2.1 fast path for evidence (MC dropout): high decisive prob + low variance
MC_FAST_PATH_CONF = float(os.environ.get("REMIX_MC_FAST_CONF", "0.8"))
MC_FAST_PATH_VAR = float(os.environ.get("REMIX_MC_FAST_VAR", "0.02"))


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


def load_veracity_checkpoint():
    """Load baseline TextClassifier or domain-adversarial checkpoint from training scripts."""
    if not MODEL_PATH.exists():
        print(f"  ⚠ Model not found at {MODEL_PATH} - using untrained TextClassifier")
        m = TextClassifier()
        return m, {}
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    sd = checkpoint.get("model_state_dict", checkpoint)
    if checkpoint.get("model_type") == "domain_adversarial" or any(
        k.startswith("veracity.") for k in sd
    ):
        nd = int(checkpoint.get("num_domains", 2))
        m = DomainAdversarialClassifier(num_domains=nd)
        m.load_state_dict(sd, strict=True)
        print(f"  ✓ Veracity model: domain-adversarial ({nd} domains)")
        return m, checkpoint
    m = TextClassifier()
    m.load_state_dict(sd, strict=False)
    print("  ✓ Veracity model: baseline DistilRoBERTa head")
    return m, checkpoint


# ============================================
# Request/Response Models
# ============================================
class DetectionRequest(BaseModel):
    text: str
    include_explanation: bool = False
    explanation_level: str = "intermediate"  # novice, intermediate, expert
    check_ai_generated: bool = False
    check_evidence: bool = False
    image_base64: Optional[str] = None
    enable_early_exit: bool = True
    # Paper §2.1: T Monte Carlo dropout passes (0 = single forward, default keeps latency)
    mc_dropout_passes: int = 0
    # When MC is on, skip evidence if decisive probability is high and uncertainty low
    use_evidence_fast_path: bool = True


class ExplainRequest(BaseModel):
    text: str
    level: str = "intermediate"  # novice, intermediate, expert


class EvidenceRequest(BaseModel):
    text: str
    max_results: int = 10
    uncertainty: float = 0.5
    depth_override: Optional[int] = None  # 5–20; Table 1 style when set


class AIDetectRequest(BaseModel):
    text: str


class ImageAnalyzeRequest(BaseModel):
    image_base64: str
    text: Optional[str] = None


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title=APP_NAME,
    description="""
    REMIX-FND: Real-time Fake News Detection System
    
    Implements paper features:
    - Module 1 (MSCIM): Text + Image analysis with adaptive fusion
    - Module 2 (EVRS): Evidence retrieval with FAISS + 100+ fact knowledge base
    - Module 3 (ELDS): AI detection with 6-detector ensemble + hierarchical explanations
    - Early Exit: Confidence-based routing for efficiency
    """,
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
explainer = None
ai_detector = None
evidence_retriever = None
image_analyzer = None
early_exit_router = None

# Background init so Render / load balancers get HTTP immediately (avoids deploy timeout).
_pipeline_ready = False
_models_loading = False
_startup_error: Optional[str] = None


def _sync_load_all():
    """Blocking ML + module init; runs in a thread pool."""
    global model, tokenizer, explainer, ai_detector, evidence_retriever, image_analyzer, early_exit_router

    print(f"\n{'='*60}")
    print(f"🚀 {APP_NAME} v{APP_VERSION} (background load)")
    print(f"   Paper-aligned routing: MC dropout optional (mc_dropout_passes); Table 1 depths; 6 AI detectors")
    print(f"{'='*60}")

    print("\n📦 Loading components...")
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    print("  ✓ Tokenizer loaded")

    model, checkpoint = load_veracity_checkpoint()
    model.eval()
    acc = checkpoint.get("val_acc", 0) if checkpoint else 0
    if acc:
        print(f"  ✓ Checkpoint val accuracy (train metrics): {acc:.1f}%")

    print("\n📚 Loading Paper Modules...")

    image_analyzer = ImageAnalyzer()
    print("  ✓ Module 1 (MSCIM): Image Analysis with manipulation detection")

    evidence_retriever = EvidenceRetriever()
    print(f"  ✓ Module 2 (EVRS): Evidence Retrieval (FAISS on-demand, {len(evidence_retriever.kb.facts)} facts)")

    ai_detector = AIContentDetector()
    print("  ✓ Module 3 (ELDS): AI Detection (6-detector ensemble)")

    explainer = HierarchicalExplainer()
    print("  ✓ Module 3 (ELDS): Hierarchical Explainer (3-tier)")

    early_exit_router = EarlyExitRouter()
    print(f"  ✓ Early Exit: Confidence-based routing (threshold: {EARLY_EXIT_THRESHOLD*100}%)")

    print(f"\n{'='*60}")
    print("✅ All features ready!")
    print(f"{'='*60}")


async def _background_load():
    global _models_loading, _pipeline_ready, _startup_error

    _models_loading = True
    _startup_error = None
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_load_all)
        _pipeline_ready = True
    except Exception as e:
        _startup_error = str(e)
        print(traceback.format_exc())
    finally:
        _models_loading = False


@app.on_event("startup")
async def startup():
    """Bind HTTP immediately; load PyTorch / HF / KB in a worker thread."""
    print("\n⚡ HTTP server up — ML pipeline loading in background (GET /health for status).")
    asyncio.create_task(_background_load())


def _require_pipeline():
    if not _pipeline_ready:
        msg = "Model pipeline is still loading. Retry shortly."
        if _startup_error:
            msg = f"Startup failed: {_startup_error}"
        raise HTTPException(status_code=503, detail=msg)


@app.get("/")
def root():
    """API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "paper_implementation": "~72%",
        "status": "running",
        "modules": {
            "MSCIM": {
                "description": "Multi-Modal Social Context Intelligence",
                "components": ["Text Encoder", "Image Analyzer", "Adaptive Fusion"],
                "status": "partial"
            },
            "EVRS": {
                "description": "Evidence-Based Verification & Retrieval",
                "components": ["FAISS Search", "100+ Facts KB", "Stance Classification"],
                "status": "active"
            },
            "ELDS": {
                "description": "Explainable LLM Detection & Defense",
                "components": ["6-Detector Ensemble", "3-Tier Explanations", "Sentence Attribution"],
                "status": "active"
            }
        },
        "optimizations": {
            "early_exit": True,
            "confidence_routing": True,
            "adaptive_depth": True
        },
        "endpoints": {
            "POST /detect": "Full detection pipeline with all features",
            "POST /explain": "3-tier hierarchical explanation",
            "POST /ai-detect": "AI content detection (6 detectors)",
            "POST /evidence": "Evidence retrieval (FAISS + KB)",
            "POST /image-analyze": "Image manipulation detection",
            "GET /health": "Health check with module status"
        }
    }


@app.get("/health")
def health():
    """Health check with module status."""
    stance_ckpt = None
    try:
        from features.evidence_retrieval_3.stance_encoder import resolve_stance_checkpoint
        p = resolve_stance_checkpoint()
        stance_ckpt = str(p) if p else None
    except Exception:
        pass
    veracity_kind = "none"
    if model is not None:
        veracity_kind = (
            "domain_adversarial"
            if model.__class__.__name__ == "DomainAdversarialClassifier"
            else "baseline"
        )
    if _startup_error:
        phase = "failed"
    elif _pipeline_ready:
        phase = "ready"
    elif _models_loading:
        phase = "loading"
    else:
        phase = "pending"

    return {
        "status": "healthy",
        "pipeline": phase,
        "ready": _pipeline_ready,
        "loading": _models_loading,
        "startup_error": _startup_error,
        "modules": {
            "text_classifier": model is not None,
            "image_analyzer": image_analyzer is not None,
            "evidence_retriever": evidence_retriever is not None,
            "ai_detector": ai_detector is not None,
            "explainer": explainer is not None,
            "early_exit": early_exit_router is not None,
        },
        "features": {
            "faiss_available": FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE,
            "knowledge_base_size": len(evidence_retriever.kb.facts) if evidence_retriever else 0,
            "ai_detectors": 6,
            "explanation_tiers": 3,
            "veracity_model": veracity_kind,
            "stance_checkpoint": stance_ckpt,
        },
    }


@app.post("/detect")
def detect(request: DetectionRequest):
    """
    Main fake news detection endpoint with all paper features.
    
    Features:
    - Text classification with DistilRoBERTa
    - Image manipulation detection (if image provided)
    - AI content detection (optional)
    - Evidence retrieval (optional)
    - 3-tier hierarchical explanations
    - Early exit for high-confidence predictions
    """
    _require_pipeline()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    start_time = time.time()
    processing_stages = []
    feature_scores = {}
    routing_info: Dict[str, Any] = {}
    
    # ========================================
    # Stage 1: Text Classification
    # ========================================
    stage_start = time.time()
    
    try:
        encoding = tokenizer(
            request.text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        T_mc = max(0, int(request.mc_dropout_passes))
        mc = predict_with_mc_dropout(
            model,
            encoding["input_ids"],
            encoding["attention_mask"],
            T=T_mc,
        )
        pred = mc["pred"]
        mean_real = mc["mean_real"]
        mean_fake = mc["mean_fake"]
        fake_prob = mean_fake * 100
        real_prob = mean_real * 100
        confidence = confidence_from_means(mean_real, mean_fake, pred)
        prediction = "FAKE" if pred == 1 else "REAL"
        routing_info = {
            "mc_dropout_passes": T_mc,
            "mean_fake_probability": mean_fake,
            "mean_real_probability": mean_real,
            "variance_fake_probability": mc["var_fake"],
            "variance_decisive_probability": mc["var_decisive"],
            "table1_depth_if_evidence": table1_depth_from_fake_variance(mc["var_fake"]),
        }
    except Exception as e:
        print(f"⚠️ Model inference error: {e}")
        # Fallback to simple heuristics
        text_lower = request.text.lower()
        suspicious = ['shocking', 'exposed', 'secret', 'conspiracy', 'breaking', 'urgent', 'bombshell']
        found = sum(1 for w in suspicious if w in text_lower)
        if found >= 2:
            prediction = "FAKE"
            confidence = 60 + found * 5
        else:
            prediction = "REAL"
            confidence = 55
        fake_prob = confidence if prediction == "FAKE" else 100 - confidence
        real_prob = 100 - fake_prob
        routing_info = {"mc_dropout_passes": 0, "fallback": True}
    
    feature_scores["text_analysis"] = confidence
    
    processing_stages.append({
        "stage": "text_classification",
        "time_ms": (time.time() - stage_start) * 1000,
        "confidence": confidence
    })
    
    # Check for early exit
    if request.enable_early_exit and confidence >= EARLY_EXIT_THRESHOLD * 100:
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "early_exit": True,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "stages_run": 1,
            "feature_scores": feature_scores,
            "routing": routing_info,
        }
        
        if request.include_explanation:
            result["explanation"] = explainer.explain(
                request.text, prediction, confidence, feature_scores, request.explanation_level
            )
        
        return to_serializable(result)
    
    # ========================================
    # Stage 2: Image Analysis (if provided)
    # ========================================
    image_result = None
    if request.image_base64:
        stage_start = time.time()
        try:
            image_data = base64.b64decode(request.image_base64)
            image_result = image_analyzer.analyze(
                image_data=image_data,
                text=request.text
            )
            feature_scores["image_analysis"] = 100 - image_result.manipulation_score
            
            # Adjust confidence based on image analysis
            if image_result.manipulation_score > 50:
                # High manipulation = less trustworthy
                if prediction == "REAL":
                    confidence = confidence * 0.8
            
            processing_stages.append({
                "stage": "image_analysis",
                "time_ms": (time.time() - stage_start) * 1000,
                "manipulation_score": image_result.manipulation_score
            })
        except Exception as e:
            print(f"Image analysis failed: {e}")
    
    # ========================================
    # Stage 3: AI Content Detection
    # ========================================
    ai_result = None
    if request.check_ai_generated:
        stage_start = time.time()
        ai_result = ai_detector.detect(request.text)
        feature_scores["ai_detection"] = ai_result["probability"]
        
        processing_stages.append({
            "stage": "ai_detection",
            "time_ms": (time.time() - stage_start) * 1000,
            "ai_probability": ai_result["probability"]
        })
    
    # ========================================
    # Stage 4: Evidence Retrieval
    # ========================================
    evidence_result = None
    if request.check_evidence:
        stage_start = time.time()
        
        T_mc = routing_info.get("mc_dropout_passes", 0) or 0
        mean_decisive = (
            routing_info.get("mean_fake_probability", fake_prob / 100.0)
            if prediction == "FAKE"
            else routing_info.get("mean_real_probability", real_prob / 100.0)
        )
        var_decisive = float(routing_info.get("variance_decisive_probability", 0.0))

        skip_evidence = (
            T_mc > 0
            and request.use_evidence_fast_path
            and evidence_fast_path(
                mean_decisive,
                var_decisive,
                conf_threshold=MC_FAST_PATH_CONF,
                var_threshold=MC_FAST_PATH_VAR,
            )
        )

        if skip_evidence:
            evidence_result = {
                "query": request.text,
                "claim_keywords": [],
                "retrieval_depth": 0,
                "search_method": "skipped",
                "evidence": [],
                "verdict": "skipped_fast_path",
                "confidence": 0.0,
                "evidence_summary": "Evidence retrieval skipped (MC routing: high decisive probability, low uncertainty).",
                "recommendation": "Enable full evidence by lowering confidence, increasing uncertainty, or set use_evidence_fast_path=false.",
            }
            feature_scores["evidence_retrieval"] = 0.0
            routing_info["evidence_fast_path"] = True
        else:
            uncertainty = 1 - (confidence / 100)
            if T_mc > 0:
                depth_override = table1_depth_from_fake_variance(
                    float(routing_info.get("variance_fake_probability", 0.0))
                )
            else:
                depth_override = None
            linear_depth = EVIDENCE_DEPTH_MIN + int(
                uncertainty * (EVIDENCE_DEPTH_MAX - EVIDENCE_DEPTH_MIN)
            )
            evidence_result = evidence_retriever.retrieve(
                request.text,
                max_results=linear_depth,
                uncertainty=uncertainty,
                depth_override=depth_override,
            )
            feature_scores["evidence_retrieval"] = evidence_result["confidence"]
            routing_info["evidence_fast_path"] = False
        
        # Adjust prediction based on evidence
        if evidence_result["verdict"] == "likely_false" and prediction == "REAL":
            confidence = confidence * 0.7
        elif evidence_result["verdict"] == "likely_true" and prediction == "FAKE":
            confidence = confidence * 0.7
        
        processing_stages.append({
            "stage": "evidence_retrieval",
            "time_ms": (time.time() - stage_start) * 1000,
            "evidence_verdict": evidence_result["verdict"]
        })
    
    # ========================================
    # Build Response
    # ========================================
    total_time = (time.time() - start_time) * 1000
    
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "early_exit": False,
        "processing_time_ms": total_time,
        "stages_run": len(processing_stages),
        "feature_scores": feature_scores,
        "processing_details": processing_stages,
        "routing": routing_info,
    }
    
    # Add image analysis results
    if image_result:
        result["image_analysis"] = {
            "manipulation_score": image_result.manipulation_score,
            "quality_score": image_result.quality_score,
            "consistency_score": image_result.consistency_score,
            "suspicious_indicators": image_result.suspicious_indicators,
            "details": image_result.details
        }
    
    # Add AI detection results
    if ai_result:
        result["ai_analysis"] = ai_result
    
    # Add evidence results
    if evidence_result:
        result["evidence"] = evidence_result
    
    # Add explanation
    if request.include_explanation:
        result["explanation"] = explainer.explain(
            request.text, 
            prediction, 
            confidence, 
            feature_scores, 
            request.explanation_level
        )
    
    # Convert all numpy types to Python native types for JSON serialization
    return to_serializable(result)


@app.post("/explain")
def explain(request: ExplainRequest):
    """
    Get 3-tier hierarchical explanation.
    
    Levels:
    - **novice**: Simple, emoji-based, actionable tips
    - **intermediate**: Technical terms, statistics, key sentences
    - **expert**: Full attribution, confidence intervals, feature weights
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    _require_pipeline()

    # Get prediction first
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
    
    confidence = max(probs[0][0].item(), probs[0][1].item()) * 100
    prediction = "FAKE" if pred == 1 else "REAL"
    
    feature_scores = {"text_analysis": confidence}
    out = explainer.explain(
        request.text, prediction, confidence, feature_scores, request.level
    )
    return to_serializable(out)


@app.post("/ai-detect")
def ai_detect(request: AIDetectRequest):
    """
    Detect if text is AI-generated using 6-detector ensemble (paper Table 2–style stack).
    
    Detectors:
    - Perplexity, burstiness, linguistic patterns, repetition, vocabulary richness,
      plus HC3 corpus retrieval similarity.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    _require_pipeline()

    return to_serializable(ai_detector.detect(request.text))


@app.post("/evidence")
def evidence(request: EvidenceRequest):
    """
    Retrieve evidence for fact-checking with FAISS.
    
    Features:
    - Semantic search (FAISS) + keyword search
    - 100+ fact knowledge base
    - Stance classification (supports/refutes/neutral)
    - Uncertainty-based adaptive depth
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    _require_pipeline()

    return to_serializable(
        evidence_retriever.retrieve(
            request.text,
            max_results=request.max_results,
            uncertainty=request.uncertainty,
            depth_override=request.depth_override,
        )
    )


@app.post("/image-analyze")
def image_analyze(request: ImageAnalyzeRequest):
    """
    Analyze image for manipulation and consistency.
    
    Checks:
    - Error Level Analysis (ELA) approximation
    - Metadata inconsistencies
    - Quality/compression analysis
    - Edge detection anomalies
    - Text-image consistency
    """
    try:
        image_data = base64.b64decode(request.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    _require_pipeline()

    result = image_analyzer.analyze(
        image_data=image_data,
        text=request.text
    )
    
    return to_serializable(
        {
            "manipulation_score": result.manipulation_score,
            "quality_score": result.quality_score,
            "consistency_score": result.consistency_score,
            "suspicious_indicators": result.suspicious_indicators,
            "metadata": result.metadata,
            "details": result.details,
        }
    )


if __name__ == "__main__":
    import uvicorn
    # Run with single worker for PyTorch compatibility
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, timeout_keep_alive=30)
