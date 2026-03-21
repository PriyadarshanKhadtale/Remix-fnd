"""
Early Exit & Confidence-Based Routing (Optimization)
=====================================================
Implements paper's multi-level optimization strategy
- Early exit for high-confidence predictions
- Adaptive module selection based on uncertainty
- Computational efficiency through smart routing
"""

import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for routing decisions."""
    VERY_HIGH = "very_high"   # > 90% - can exit early
    HIGH = "high"             # 75-90% - minimal additional processing
    MEDIUM = "medium"         # 50-75% - standard processing
    LOW = "low"               # < 50% - full pipeline needed


@dataclass
class ModuleResult:
    """Result from a single module."""
    module_name: str
    prediction: str
    confidence: float
    processing_time: float
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision on how to route the analysis."""
    should_exit: bool
    next_modules: List[str]
    confidence_level: ConfidenceLevel
    reason: str


class EarlyExitRouter:
    """
    Manages early exit and routing decisions.
    
    Paper implementation:
    - Exit at layer L if confidence > τ_L (threshold)
    - Use lightweight modules first
    - Progressively add expensive modules only if needed
    """
    
    # Confidence thresholds for early exit at each stage
    THRESHOLDS = {
        "stage_1_text": 0.92,      # Text-only analysis
        "stage_2_evidence": 0.85,  # + Evidence retrieval
        "stage_3_ai": 0.80,        # + AI detection
        "stage_4_image": 0.75,     # + Image analysis
    }
    
    # Module costs (relative computational cost)
    MODULE_COSTS = {
        "text_analysis": 1.0,      # Baseline
        "evidence_retrieval": 3.0,  # FAISS search
        "ai_detection": 2.0,        # Multiple detectors
        "image_analysis": 4.0,      # Image processing
        "explanation": 1.5,         # Generation
    }
    
    # Processing order (cheapest first)
    MODULE_ORDER = [
        "text_analysis",
        "ai_detection",
        "evidence_retrieval",
        "image_analysis",
        "explanation"
    ]
    
    def __init__(self, enable_early_exit: bool = True):
        self.enable_early_exit = enable_early_exit
        self.processing_history = []
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence score to level."""
        if confidence >= 90:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 75:
            return ConfidenceLevel.HIGH
        elif confidence >= 50:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def should_exit_early(
        self,
        current_stage: str,
        confidence: float,
        processed_modules: List[str]
    ) -> RoutingDecision:
        """
        Determine if we should exit early or continue processing.
        
        Args:
            current_stage: Current processing stage
            confidence: Current confidence level (0-100)
            processed_modules: List of already processed modules
        """
        if not self.enable_early_exit:
            # No early exit - continue with all remaining modules
            remaining = [m for m in self.MODULE_ORDER if m not in processed_modules]
            return RoutingDecision(
                should_exit=len(remaining) == 0,
                next_modules=remaining[:1] if remaining else [],
                confidence_level=self.get_confidence_level(confidence),
                reason="Early exit disabled - running full pipeline"
            )
        
        confidence_normalized = confidence / 100.0
        threshold = self.THRESHOLDS.get(current_stage, 0.85)
        confidence_level = self.get_confidence_level(confidence)
        
        # Very high confidence - exit early
        if confidence_normalized >= threshold:
            return RoutingDecision(
                should_exit=True,
                next_modules=[],
                confidence_level=confidence_level,
                reason=f"Early exit: confidence {confidence:.1f}% >= threshold {threshold*100:.0f}%"
            )
        
        # Determine next modules based on confidence level
        remaining = [m for m in self.MODULE_ORDER if m not in processed_modules]
        
        if confidence_level == ConfidenceLevel.HIGH:
            # Only add one more module
            next_modules = remaining[:1] if remaining else []
            reason = "High confidence - adding one additional check"
        elif confidence_level == ConfidenceLevel.MEDIUM:
            # Add two more modules
            next_modules = remaining[:2] if remaining else []
            reason = "Medium confidence - running additional analysis"
        else:
            # Low confidence - run all remaining
            next_modules = remaining
            reason = "Low confidence - running full pipeline"
        
        return RoutingDecision(
            should_exit=len(next_modules) == 0,
            next_modules=next_modules,
            confidence_level=confidence_level,
            reason=reason
        )
    
    def calculate_adaptive_depth(self, confidence: float) -> Dict[str, int]:
        """
        Calculate adaptive depth/parameters for each module.
        
        Paper: Uncertainty-based depth control for evidence retrieval
        """
        uncertainty = 1.0 - (confidence / 100.0)
        
        return {
            "evidence_retrieval_depth": 5 + int(uncertainty * 15),  # 5-20 documents
            "ai_detection_detectors": 3 + int(uncertainty * 3),     # 3-6 detectors
            "explanation_detail_level": "expert" if uncertainty > 0.4 else "intermediate"
        }
    
    def estimate_remaining_cost(self, processed_modules: List[str]) -> float:
        """Estimate computational cost of remaining modules."""
        remaining = [m for m in self.MODULE_ORDER if m not in processed_modules]
        return sum(self.MODULE_COSTS.get(m, 1.0) for m in remaining)
    
    def log_decision(
        self,
        stage: str,
        decision: RoutingDecision,
        confidence: float,
        elapsed_time: float
    ):
        """Log routing decision for analysis."""
        self.processing_history.append({
            "stage": stage,
            "confidence": confidence,
            "decision": decision.reason,
            "exited": decision.should_exit,
            "time_ms": elapsed_time * 1000
        })
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing decisions."""
        if not self.processing_history:
            return {"stages": 0, "total_time_ms": 0}
        
        return {
            "stages": len(self.processing_history),
            "total_time_ms": sum(h["time_ms"] for h in self.processing_history),
            "final_confidence": self.processing_history[-1]["confidence"],
            "early_exit": self.processing_history[-1]["exited"],
            "history": self.processing_history
        }
    
    def reset(self):
        """Reset processing history."""
        self.processing_history = []


class AdaptivePipeline:
    """
    Orchestrates the full analysis pipeline with early exit support.
    """
    
    def __init__(self, enable_early_exit: bool = True):
        self.router = EarlyExitRouter(enable_early_exit)
        self.modules = {}
        self.results = []
    
    def register_module(self, name: str, func: Callable, cost: float = 1.0):
        """Register a processing module."""
        self.modules[name] = {
            "func": func,
            "cost": cost
        }
        self.router.MODULE_COSTS[name] = cost
    
    def run(
        self,
        text: str,
        image_data: Optional[bytes] = None,
        force_full: bool = False
    ) -> Dict[str, Any]:
        """
        Run the adaptive pipeline.
        
        Args:
            text: Input text to analyze
            image_data: Optional image data
            force_full: Force full pipeline (no early exit)
        """
        self.router.reset()
        self.results = []
        
        processed_modules = []
        current_confidence = 50.0
        current_prediction = "UNCERTAIN"
        all_features = {}
        
        start_time = time.time()
        
        # Stage 1: Text Analysis (always run)
        if "text_analysis" in self.modules:
            stage_start = time.time()
            result = self.modules["text_analysis"]["func"](text)
            stage_time = time.time() - stage_start
            
            processed_modules.append("text_analysis")
            current_confidence = result.get("confidence", 50)
            current_prediction = result.get("prediction", "UNCERTAIN")
            all_features["text_analysis"] = result.get("confidence", 50)
            
            self.results.append(ModuleResult(
                module_name="text_analysis",
                prediction=current_prediction,
                confidence=current_confidence,
                processing_time=stage_time,
                features=result
            ))
            
            # Check for early exit
            decision = self.router.should_exit_early(
                "stage_1_text", current_confidence, processed_modules
            )
            self.router.log_decision("stage_1_text", decision, current_confidence, stage_time)
            
            if decision.should_exit and not force_full:
                return self._build_response(
                    current_prediction, current_confidence, all_features, start_time
                )
        
        # Stage 2+: Dynamic module execution
        max_iterations = len(self.router.MODULE_ORDER)
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            decision = self.router.should_exit_early(
                f"stage_{iteration + 1}",
                current_confidence,
                processed_modules
            )
            
            if decision.should_exit and not force_full:
                break
            
            if not decision.next_modules:
                break
            
            for module_name in decision.next_modules:
                if module_name not in self.modules:
                    continue
                
                stage_start = time.time()
                
                # Special handling for image analysis
                if module_name == "image_analysis" and image_data is None:
                    processed_modules.append(module_name)
                    continue
                
                # Run module
                try:
                    if module_name == "image_analysis":
                        result = self.modules[module_name]["func"](image_data, text)
                    elif module_name == "evidence_retrieval":
                        depth = self.router.calculate_adaptive_depth(current_confidence)
                        result = self.modules[module_name]["func"](
                            text, depth["evidence_retrieval_depth"]
                        )
                    else:
                        result = self.modules[module_name]["func"](text)
                    
                    stage_time = time.time() - stage_start
                    processed_modules.append(module_name)
                    
                    # Update confidence (weighted average)
                    module_confidence = result.get("confidence", 50)
                    if isinstance(module_confidence, (int, float)):
                        current_confidence = (current_confidence * 0.6 + module_confidence * 0.4)
                    
                    all_features[module_name] = module_confidence
                    
                    self.results.append(ModuleResult(
                        module_name=module_name,
                        prediction=result.get("prediction", current_prediction),
                        confidence=module_confidence,
                        processing_time=stage_time,
                        features=result
                    ))
                    
                    self.router.log_decision(
                        f"stage_{len(processed_modules)}",
                        decision,
                        current_confidence,
                        stage_time
                    )
                    
                except Exception as e:
                    processed_modules.append(module_name)
                    print(f"Module {module_name} failed: {e}")
        
        return self._build_response(
            current_prediction, current_confidence, all_features, start_time
        )
    
    def _build_response(
        self,
        prediction: str,
        confidence: float,
        features: Dict[str, float],
        start_time: float
    ) -> Dict[str, Any]:
        """Build final response."""
        total_time = time.time() - start_time
        summary = self.router.get_processing_summary()
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "feature_scores": features,
            "processing": {
                "total_time_ms": total_time * 1000,
                "modules_run": [r.module_name for r in self.results],
                "early_exit": summary.get("early_exit", False),
                "stages": summary.get("stages", 0)
            },
            "module_results": [
                {
                    "name": r.module_name,
                    "confidence": r.confidence,
                    "time_ms": r.processing_time * 1000
                }
                for r in self.results
            ],
            "efficiency": {
                "modules_saved": len(self.router.MODULE_ORDER) - len(self.results),
                "estimated_cost_saved": self.router.estimate_remaining_cost(
                    [r.module_name for r in self.results]
                )
            }
        }


# Singleton pipeline
_pipeline = None

def get_pipeline(enable_early_exit: bool = True) -> AdaptivePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AdaptivePipeline(enable_early_exit)
    return _pipeline

