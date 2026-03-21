"""
Utility Functions
=================
Helper functions used across the backend.
"""

import torch
import time
from functools import wraps
from typing import Callable, Any


def get_device() -> str:
    """
    Get the best available device for PyTorch.
    
    Returns:
        'cuda' if GPU available, 'mps' for Apple Silicon, else 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()

