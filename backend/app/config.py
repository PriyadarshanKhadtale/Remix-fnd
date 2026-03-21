"""
Configuration Settings
======================
All configuration for the REMIX-FND backend.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App Info
    APP_NAME: str = "REMIX-FND"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    
    # Model Settings
    TEXT_MODEL_NAME: str = "distilroberta-base"
    TEXT_MODEL_PATH: Optional[str] = None
    MAX_TEXT_LENGTH: int = 128
    
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

