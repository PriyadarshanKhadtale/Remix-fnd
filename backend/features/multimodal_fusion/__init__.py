"""Learned attention fusion over text, image, social, and temporal signals."""

from .fusion import MultimodalFusion, fuse_detection_signals

__all__ = ["MultimodalFusion", "fuse_detection_signals"]
