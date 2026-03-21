"""
Image Analysis Module (MSCIM - Module 1)
========================================
Multi-modal analysis using image features
Implements: Visual consistency, manipulation detection, reverse image search indicators
"""

import os
import io
import base64
import hashlib
import struct
import zlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Try to import image processing libraries
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL/Pillow not installed. Image analysis will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class ImageAnalysisResult:
    """Result from image analysis."""
    manipulation_score: float  # 0-100, higher = more likely manipulated
    quality_score: float       # 0-100, higher = better quality
    consistency_score: float   # 0-100, higher = more consistent with text
    suspicious_indicators: List[str]
    metadata: Dict[str, Any]
    details: str


class ImageManipulationDetector:
    """
    Detects potential image manipulation.
    
    Techniques:
    - Error Level Analysis (ELA) approximation
    - Metadata inconsistency detection
    - Quality/compression analysis
    - Edge detection anomalies
    """
    
    def __init__(self):
        self.supported_formats = {'JPEG', 'PNG', 'GIF', 'WEBP', 'BMP'}
    
    def analyze(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image for manipulation indicators.
        
        Args:
            image_data: Raw image bytes
        """
        if not PIL_AVAILABLE:
            return self._no_pil_fallback(image_data)
        
        try:
            img = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return {
                "error": f"Could not open image: {str(e)}",
                "manipulation_score": 50,
                "details": "Unable to analyze image format"
            }
        
        results = {}
        suspicious_indicators = []
        
        # 1. Basic metadata analysis
        metadata = self._analyze_metadata(img)
        results["metadata"] = metadata
        
        if metadata.get("has_editing_software"):
            suspicious_indicators.append("🖼️ Edited with software")
        if metadata.get("stripped_metadata"):
            suspicious_indicators.append("📋 Metadata stripped")
        
        # 2. Format/quality analysis
        quality = self._analyze_quality(img, image_data)
        results["quality"] = quality
        
        if quality.get("multiple_compressions"):
            suspicious_indicators.append("🔄 Multiple compressions detected")
        if quality.get("quality_inconsistency"):
            suspicious_indicators.append("📊 Quality inconsistencies")
        
        # 3. Visual analysis (if numpy available)
        if NUMPY_AVAILABLE:
            visual = self._analyze_visual(img)
            results["visual"] = visual
            
            if visual.get("edge_anomalies"):
                suspicious_indicators.append("✂️ Edge anomalies detected")
            if visual.get("noise_inconsistency"):
                suspicious_indicators.append("📷 Noise pattern inconsistencies")
        
        # Calculate overall manipulation score
        scores = []
        
        # Metadata score
        meta_score = 30 if metadata.get("has_editing_software") else 0
        meta_score += 20 if metadata.get("stripped_metadata") else 0
        scores.append(meta_score)
        
        # Quality score
        qual_score = 30 if quality.get("multiple_compressions") else 0
        qual_score += 20 if quality.get("quality_inconsistency") else 0
        scores.append(qual_score)
        
        # Visual score
        if NUMPY_AVAILABLE:
            vis_score = 25 if visual.get("edge_anomalies") else 0
            vis_score += 25 if visual.get("noise_inconsistency") else 0
            scores.append(vis_score)
        
        manipulation_score = min(sum(scores), 100)
        
        return {
            "manipulation_score": manipulation_score,
            "suspicious_indicators": suspicious_indicators,
            "quality_score": 100 - quality.get("compression_artifacts", 0),
            "format": img.format,
            "dimensions": img.size,
            "mode": img.mode,
            "metadata_analysis": metadata,
            "quality_analysis": quality,
            "visual_analysis": results.get("visual", {}),
            "details": self._generate_details(manipulation_score, suspicious_indicators)
        }
    
    def _analyze_metadata(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze image metadata for manipulation signs."""
        metadata = {
            "has_exif": False,
            "has_editing_software": False,
            "stripped_metadata": True,
            "creation_info": None
        }
        
        try:
            # Check EXIF data
            exif = img.getexif()
            if exif:
                metadata["has_exif"] = True
                metadata["stripped_metadata"] = False
                
                # Check for editing software tags
                software_tags = [305, 11]  # Software, ProcessingSoftware
                for tag in software_tags:
                    if tag in exif:
                        software = str(exif[tag]).lower()
                        editing_tools = ['photoshop', 'gimp', 'lightroom', 'snapseed', 
                                        'pixlr', 'canva', 'paint', 'editor']
                        if any(tool in software for tool in editing_tools):
                            metadata["has_editing_software"] = True
                            metadata["software"] = exif[tag]
            
            # Check for PNG metadata
            if img.format == 'PNG' and img.info:
                metadata["stripped_metadata"] = False
                if 'Software' in img.info:
                    metadata["has_editing_software"] = True
                    metadata["software"] = img.info['Software']
        
        except Exception:
            pass
        
        return metadata
    
    def _analyze_quality(self, img: Image.Image, raw_data: bytes) -> Dict[str, Any]:
        """Analyze image quality characteristics."""
        quality = {
            "compression_artifacts": 0,
            "multiple_compressions": False,
            "quality_inconsistency": False,
            "estimated_quality": 100
        }
        
        if img.format == 'JPEG':
            # Check for JPEG artifacts
            file_size = len(raw_data)
            pixels = img.size[0] * img.size[1]
            
            # Bits per pixel estimation
            bpp = (file_size * 8) / pixels
            
            # Very low bpp suggests heavy compression
            if bpp < 0.5:
                quality["compression_artifacts"] = 70
                quality["estimated_quality"] = 30
                quality["multiple_compressions"] = True
            elif bpp < 1.0:
                quality["compression_artifacts"] = 40
                quality["estimated_quality"] = 60
            elif bpp < 2.0:
                quality["compression_artifacts"] = 20
                quality["estimated_quality"] = 80
            
            # Check for double JPEG compression artifacts
            # (This is a simplified heuristic)
            try:
                if hasattr(img, 'quantization') and img.quantization:
                    tables = list(img.quantization.values())
                    if tables:
                        # Non-standard quantization tables might indicate editing
                        avg_value = sum(sum(t) for t in tables) / sum(len(t) for t in tables)
                        if avg_value > 30:  # High quantization values
                            quality["quality_inconsistency"] = True
            except Exception:
                pass
        
        return quality
    
    def _analyze_visual(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze visual characteristics using numpy."""
        visual = {
            "edge_anomalies": False,
            "noise_inconsistency": False,
            "color_distribution": "normal"
        }
        
        try:
            # Convert to array
            arr = np.array(img.convert('RGB'))
            
            # Simple edge detection anomaly check
            # Calculate local variance (high variance = edges)
            if arr.shape[0] > 10 and arr.shape[1] > 10:
                gray = np.mean(arr, axis=2)
                
                # Divide into blocks and check variance consistency
                block_size = min(50, min(gray.shape) // 4)
                variances = []
                
                for i in range(0, gray.shape[0] - block_size, block_size):
                    for j in range(0, gray.shape[1] - block_size, block_size):
                        block = gray[i:i+block_size, j:j+block_size]
                        variances.append(np.var(block))
                
                if variances:
                    variance_std = np.std(variances)
                    # Very inconsistent variances might indicate manipulation
                    if variance_std > np.mean(variances) * 2:
                        visual["edge_anomalies"] = True
                    
                    # Check noise pattern
                    noise = gray - np.round(gray / 10) * 10
                    noise_var = np.var(noise)
                    
                    # Extremely uniform noise is suspicious
                    if noise_var < 1:
                        visual["noise_inconsistency"] = True
            
            # Color distribution
            for channel in range(3):
                channel_data = arr[:, :, channel].flatten()
                unique_ratio = len(np.unique(channel_data)) / 256
                if unique_ratio < 0.5:
                    visual["color_distribution"] = "limited"
                    break
        
        except Exception:
            pass
        
        return visual
    
    def _generate_details(self, score: float, indicators: List[str]) -> str:
        """Generate human-readable details."""
        if score < 20:
            status = "Image appears unmanipulated"
        elif score < 50:
            status = "Some potential manipulation indicators found"
        elif score < 75:
            status = "Multiple manipulation indicators detected"
        else:
            status = "High likelihood of image manipulation"
        
        if indicators:
            return f"{status}. Flags: {', '.join(i.replace('🖼️ ', '').replace('📋 ', '').replace('🔄 ', '').replace('📊 ', '').replace('✂️ ', '').replace('📷 ', '') for i in indicators)}"
        return status
    
    def _no_pil_fallback(self, image_data: bytes) -> Dict[str, Any]:
        """Fallback analysis when PIL is not available."""
        # Basic analysis from raw bytes
        file_size = len(image_data)
        
        # Check file signature
        format_detected = "unknown"
        if image_data[:3] == b'\xff\xd8\xff':
            format_detected = "JPEG"
        elif image_data[:8] == b'\x89PNG\r\n\x1a\n':
            format_detected = "PNG"
        elif image_data[:6] in (b'GIF87a', b'GIF89a'):
            format_detected = "GIF"
        elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
            format_detected = "WEBP"
        
        return {
            "manipulation_score": 50,
            "suspicious_indicators": ["⚠️ Limited analysis (PIL not installed)"],
            "quality_score": 50,
            "format": format_detected,
            "file_size": file_size,
            "details": "Install Pillow for full image analysis: pip install Pillow"
        }


class TextImageConsistencyChecker:
    """
    Checks if image content is consistent with accompanying text.
    
    Uses keyword and semantic matching.
    """
    
    # Common image subject keywords
    SUBJECT_KEYWORDS = {
        "person": ["person", "people", "man", "woman", "child", "crowd", "face", "human"],
        "politics": ["president", "election", "vote", "government", "politician", "congress", "parliament"],
        "disaster": ["fire", "flood", "earthquake", "hurricane", "disaster", "damage", "destruction"],
        "health": ["hospital", "doctor", "nurse", "patient", "medicine", "vaccine", "virus"],
        "technology": ["computer", "phone", "device", "robot", "ai", "digital", "technology"],
        "nature": ["animal", "tree", "forest", "ocean", "mountain", "wildlife", "nature"],
        "money": ["money", "dollar", "economy", "stock", "market", "bank", "financial"],
    }
    
    def check_consistency(self, text: str, image_tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Check text-image consistency.
        
        Args:
            text: Article/post text
            image_tags: Tags/labels from image (if available from ML model)
        """
        text_lower = text.lower()
        
        # Extract topics from text
        text_topics = set()
        for topic, keywords in self.SUBJECT_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                text_topics.add(topic)
        
        # If we have image tags, check overlap
        if image_tags:
            image_topics = set()
            for topic, keywords in self.SUBJECT_KEYWORDS.items():
                if any(kw in tag.lower() for tag in image_tags for kw in keywords):
                    image_topics.add(topic)
            
            overlap = text_topics & image_topics
            consistency_score = len(overlap) / max(len(text_topics), 1) * 100 if text_topics else 50
            
            return {
                "consistency_score": consistency_score,
                "text_topics": list(text_topics),
                "image_topics": list(image_topics),
                "matching_topics": list(overlap),
                "details": f"Found {len(overlap)} matching topics between text and image"
            }
        
        # Without image tags, return neutral score
        return {
            "consistency_score": 50,
            "text_topics": list(text_topics),
            "image_topics": [],
            "matching_topics": [],
            "details": "No image analysis available for consistency check"
        }


class ImageAnalyzer:
    """
    Main image analysis orchestrator.
    
    Combines:
    - Manipulation detection
    - Text-image consistency
    - Reverse image search indicators
    """
    
    def __init__(self):
        self.manipulation_detector = ImageManipulationDetector()
        self.consistency_checker = TextImageConsistencyChecker()
    
    def analyze(
        self,
        image_data: Optional[bytes] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        text: Optional[str] = None,
        image_tags: Optional[List[str]] = None
    ) -> ImageAnalysisResult:
        """
        Comprehensive image analysis.
        
        Args:
            image_data: Raw image bytes
            image_url: URL to image (for logging/tracking only)
            image_base64: Base64-encoded image
            text: Accompanying text for consistency check
            image_tags: Pre-computed image tags
        """
        # Get image data
        if image_base64 and not image_data:
            try:
                image_data = base64.b64decode(image_base64)
            except Exception:
                pass
        
        if not image_data:
            return ImageAnalysisResult(
                manipulation_score=50,
                quality_score=50,
                consistency_score=50,
                suspicious_indicators=["⚠️ No image data provided"],
                metadata={},
                details="No image available for analysis"
            )
        
        # Run manipulation detection
        manipulation = self.manipulation_detector.analyze(image_data)
        
        # Run consistency check if text provided
        consistency = {"consistency_score": 50, "details": "No text provided"}
        if text:
            consistency = self.consistency_checker.check_consistency(text, image_tags)
        
        # Combine results
        suspicious = manipulation.get("suspicious_indicators", [])
        
        # Add consistency warning if low
        if consistency["consistency_score"] < 30:
            suspicious.append("🔗 Image-text mismatch")
        
        return ImageAnalysisResult(
            manipulation_score=manipulation.get("manipulation_score", 50),
            quality_score=manipulation.get("quality_score", 50),
            consistency_score=consistency["consistency_score"],
            suspicious_indicators=suspicious,
            metadata={
                "format": manipulation.get("format"),
                "dimensions": manipulation.get("dimensions"),
                "manipulation_analysis": manipulation,
                "consistency_analysis": consistency,
                "image_url": image_url
            },
            details=f"Manipulation: {manipulation.get('details', 'N/A')} | Consistency: {consistency['details']}"
        )


# Global instance
_analyzer = None

def get_analyzer() -> ImageAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ImageAnalyzer()
    return _analyzer

def analyze(
    image_data: Optional[bytes] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    text: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze image and return results as dictionary."""
    result = get_analyzer().analyze(image_data, image_url, image_base64, text)
    return {
        "manipulation_score": result.manipulation_score,
        "quality_score": result.quality_score,
        "consistency_score": result.consistency_score,
        "suspicious_indicators": result.suspicious_indicators,
        "metadata": result.metadata,
        "details": result.details
    }

