"""
Text Predictor
==============
High-level prediction interface for text analysis.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from core.veracity_checkpoint import load_env_files, resolve_veracity_model_path
from .model import TextClassifier
from .preprocessor import TextPreprocessor


class TextPredictor:
    """
    Main predictor class for text-based fake news detection.
    
    Usage:
        predictor = TextPredictor()
        predictor.load_model("path/to/checkpoint.pt")
        result = predictor.predict("Some news text")
    """
    
    def __init__(self, model_name: str = "distilroberta-base", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        
        # Initialize components
        self.preprocessor = TextPreprocessor(model_name=model_name)
        self.model: Optional[TextClassifier] = None
        self._is_loaded = False
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load trained model from checkpoint."""
        self.model = TextClassifier(model_name=self.model_name)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self._is_loaded = True
        print(f"✓ Text model loaded from {checkpoint_path}")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if news is fake or real.
        
        Args:
            text: News headline or article text
            
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        inputs = self.preprocessor.tokenize(text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
        
        fake_prob = probs[0][1].item() * 100
        real_prob = probs[0][0].item() * 100
        
        return {
            "prediction": "FAKE" if pred == 1 else "REAL",
            "confidence": max(fake_prob, real_prob),
            "fake_probability": fake_prob,
            "real_probability": real_prob
        }
    
    def get_attention(self, text: str) -> torch.Tensor:
        """Get attention weights for explainability."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.preprocessor.tokenize(text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            attention = self.model.get_attention_weights(input_ids, attention_mask)
        
        return attention


# Global predictor instance (singleton pattern)
_predictor: Optional[TextPredictor] = None


def get_predictor() -> TextPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    
    if _predictor is None:
        _predictor = TextPredictor()
        backend_dir = Path(__file__).resolve().parents[2]
        repo_root = Path(__file__).resolve().parents[3]
        load_env_files(backend_dir / ".env", repo_root / ".env", override=False)

        to_load: Optional[Path] = None
        explicit = os.environ.get("TEXT_MODEL_PATH", "").strip()
        if explicit:
            ep = Path(explicit)
            if ep.is_file():
                to_load = ep
            else:
                print(f"⚠ TextPredictor: TEXT_MODEL_PATH not found: {ep}")
        if to_load is None:
            cand = resolve_veracity_model_path(repo_root)
            if cand.is_file():
                to_load = cand
        if to_load is None:
            fb = repo_root.parent / "models" / "text_classifier" / "best_model.pt"
            if fb.is_file():
                to_load = fb

        if to_load is not None:
            try:
                _predictor.load_model(str(to_load))
            except Exception as e:
                print(f"⚠ TextPredictor: could not load {to_load}: {e}")

    return _predictor


def predict(text: str) -> Dict[str, Any]:
    """Convenience function for quick predictions."""
    return get_predictor().predict(text)

