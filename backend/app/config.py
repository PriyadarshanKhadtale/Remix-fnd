"""
Configuration Settings
======================
Settings for the **modular** FastAPI app (`uvicorn app.main:app`).

The **full paper-aligned orchestration** (MC dropout, multimodal `/detect`, DSRG, early exit)
lives in `run.py`. Env vars below mirror **SCOPE.md** and **docs/ARCHITECTURE.md**; `run.py` and
`core.veracity_checkpoint` read the same names from the environment.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App Info
    APP_NAME: str = "REMIX-FND"
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths — REMIX_FND_v2 repo root (parent of backend/)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Model Settings
    TEXT_MODEL_NAME: str = "distilroberta-base"
    TEXT_MODEL_PATH: Optional[str] = None
    MAX_TEXT_LENGTH: int = 128

    # Veracity checkpoint (same semantics as run.py / SCOPE.md)
    REMIX_VERACITY_CKPT: Optional[str] = None
    REMIX_VERACITY_RUN_ID: Optional[str] = None
    REMIX_VERACITY_VARIANT: str = "dann"

    # MC evidence fast path (run.py only; documented here for .env parity)
    REMIX_MC_FAST_CONF: float = 0.8
    REMIX_MC_FAST_VAR: float = 0.02
    
    # Feature Toggles (enable/disable features)
    ENABLE_TEXT_ANALYSIS: bool = True
    ENABLE_IMAGE_ANALYSIS: bool = False
    ENABLE_EVIDENCE_RETRIEVAL: bool = False
    ENABLE_AI_DETECTION: bool = False
    ENABLE_EXPLAINABILITY: bool = True
    
    # Device
    DEVICE: str = "cpu"  # cpu, cuda, mps
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

