"""
Health Check Endpoint
=====================
Monitor API health and status.
"""

from fastapi import APIRouter
from ..config import settings

router = APIRouter()


@router.get("/health")
def health_check():
    """Check if the API is running and healthy."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION
    }

