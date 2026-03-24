"""
Evidence Retrieval Endpoint
===========================
Retrieve supporting/contradicting evidence for claims.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ..config import settings

router = APIRouter()


class EvidenceRequest(BaseModel):
    """Request body for evidence retrieval."""
    text: str
    max_results: int = 5


class EvidenceItem(BaseModel):
    """A single piece of evidence."""
    source: str
    title: str
    snippet: str
    relevance_score: float
    supports_claim: Optional[bool] = None  # True=supports, False=contradicts, None=neutral


class EvidenceResponse(BaseModel):
    """Evidence retrieval response."""
    query: str
    evidence: List[EvidenceItem]
    verdict: str  # "supported", "contradicted", "inconclusive"
    confidence: float


@router.post("/evidence", response_model=EvidenceResponse)
async def retrieve_evidence(request: EvidenceRequest):
    """
    Retrieve evidence related to the news claim.
    
    - **text**: The claim to fact-check
    - **max_results**: Maximum number of evidence items to return
    """
    if not settings.ENABLE_EVIDENCE_RETRIEVAL:
        raise HTTPException(status_code=503, detail="Evidence retrieval feature is disabled")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    from features.evidence_retrieval_3 import retriever
    
    result = retriever.retrieve(
        request.text,
        max_results=request.max_results
    )
    
    return result

