"""
REMIX-FND Backend API
=====================
Main FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import detect, explain, evidence, health


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Real-time Fake News Detection System with Explainability",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(detect.router, tags=["Detection"])
app.include_router(explain.router, tags=["Explainability"])
app.include_router(evidence.router, tags=["Evidence"])


@app.get("/")
def root():
    """API root - shows available endpoints."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "POST /detect": "Analyze news for fake content",
            "POST /explain": "Get explanation for detection",
            "POST /evidence": "Retrieve supporting evidence",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        },
        "features": {
            "text_analysis": settings.ENABLE_TEXT_ANALYSIS,
            "image_analysis": settings.ENABLE_IMAGE_ANALYSIS,
            "evidence_retrieval": settings.ENABLE_EVIDENCE_RETRIEVAL,
            "ai_detection": settings.ENABLE_AI_DETECTION,
            "explainability": settings.ENABLE_EXPLAINABILITY
        }
    }


# Startup event
@app.on_event("startup")
async def startup():
    """Load models on startup."""
    print(f"\n{'='*50}")
    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"{'='*50}")
    print(f"Features enabled:")
    print(f"  ✓ Text Analysis: {settings.ENABLE_TEXT_ANALYSIS}")
    print(f"  {'✓' if settings.ENABLE_IMAGE_ANALYSIS else '○'} Image Analysis: {settings.ENABLE_IMAGE_ANALYSIS}")
    print(f"  {'✓' if settings.ENABLE_EVIDENCE_RETRIEVAL else '○'} Evidence Retrieval: {settings.ENABLE_EVIDENCE_RETRIEVAL}")
    print(f"  {'✓' if settings.ENABLE_AI_DETECTION else '○'} AI Detection: {settings.ENABLE_AI_DETECTION}")
    print(f"  {'✓' if settings.ENABLE_EXPLAINABILITY else '○'} Explainability: {settings.ENABLE_EXPLAINABILITY}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

