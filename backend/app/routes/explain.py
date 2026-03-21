"""
Explainability Endpoint
=======================
Get detailed explanations for predictions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..config import settings

router = APIRouter()


class ExplainRequest(BaseModel):
    """Request body for explanation."""
    text: str
    detail_level: str = "simple"  # simple, detailed, expert


class HighlightedWord(BaseModel):
    """A word with its importance score."""
    word: str
    importance: float
    is_suspicious: bool


class ExplanationResponse(BaseModel):
    """Detailed explanation response."""
    summary: str
    key_factors: List[str]
    highlighted_words: List[HighlightedWord]
    confidence_breakdown: dict
    suggestions: List[str]


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplainRequest):
    """
    Get detailed explanation for why news was classified as fake/real.
    
    - **text**: The news text to explain
    - **detail_level**: How detailed the explanation should be
    """
    if not settings.ENABLE_EXPLAINABILITY:
        raise HTTPException(status_code=503, detail="Explainability feature is disabled")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    from ...features.explainability_5 import explainer
    from ...features.text_analysis_1 import predictor
    
    # Get prediction first
    prediction = predictor.predict(request.text)
    
    # Get explanation
    explanation = explainer.get_detailed_explanation(
        request.text, 
        prediction,
        detail_level=request.detail_level
    )
    
    return explanation

