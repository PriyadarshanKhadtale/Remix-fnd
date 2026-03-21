"""
Detection Endpoint
==================
Main endpoint for fake news detection.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..config import settings
from ...features.text_analysis_1 import predictor as text_predictor
from ...features.ai_detection_4 import detector as ai_detector

router = APIRouter()


class DetectionRequest(BaseModel):
    """Request body for detection."""
    text: str
    include_explanation: bool = False
    check_ai_generated: bool = False


class DetectionResponse(BaseModel):
    """Response from detection."""
    prediction: str  # "REAL" or "FAKE"
    confidence: float
    fake_probability: float
    real_probability: float
    ai_generated: Optional[dict] = None
    explanation: Optional[dict] = None


@router.post("/detect", response_model=DetectionResponse)
async def detect_fake_news(request: DetectionRequest):
    """
    Detect if news is fake or real.
    
    - **text**: The news headline or article to analyze
    - **include_explanation**: Get explanation for the prediction
    - **check_ai_generated**: Check if text is AI-generated
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Main text analysis
    result = text_predictor.predict(request.text)
    
    response = DetectionResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        fake_probability=result["fake_probability"],
        real_probability=result["real_probability"]
    )
    
    # Optional: AI detection
    if request.check_ai_generated and settings.ENABLE_AI_DETECTION:
        response.ai_generated = ai_detector.detect(request.text)
    
    # Optional: Explanation
    if request.include_explanation and settings.ENABLE_EXPLAINABILITY:
        from ...features.explainability_5 import explainer
        response.explanation = explainer.explain(request.text, result)
    
    return response

