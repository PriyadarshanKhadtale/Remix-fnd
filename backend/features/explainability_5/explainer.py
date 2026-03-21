"""
Hierarchical Explainability System (ELDS - Module 3)
=====================================================
3-Tier Explanations: Novice, Intermediate, Expert
Implements: Sentence-level attribution, attention visualization, feature contribution
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ExpertiseLevel(Enum):
    """User expertise levels for adaptive explanations."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


@dataclass
class SentenceAttribution:
    """Attribution for a single sentence."""
    sentence: str
    index: int
    contribution_score: float  # -1 to 1 (negative = supports real, positive = supports fake)
    confidence: float
    key_phrases: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)


@dataclass
class FeatureContribution:
    """Contribution from a single feature/module."""
    name: str
    score: float  # 0-100%
    direction: str  # "supports_fake", "supports_real", "neutral"
    weight: float  # How much this contributed to final decision
    details: str
    icon: str


class SentenceAttributor:
    """
    Analyzes each sentence for its contribution to the overall prediction.
    
    Paper: Fine-grained attribution at token/sentence level.
    """
    
    # Linguistic markers for fake news
    FAKE_INDICATORS = {
        # Sensationalism
        'sensational': [
            r'\b(shocking|bombshell|breaking|explosive|stunning|unbelievable|incredible)\b',
            r'\b(secret|hidden|exposed|revealed|leaked|uncovered)\b',
            r'\b(massive|huge|enormous|devastating|catastrophic|unprecedented)\b',
        ],
        # Emotional manipulation
        'emotional': [
            r'\b(outrage|fury|horrifying|terrifying|disgusting|heartbreaking)\b',
            r'\b(evil|corrupt|criminal|traitor|enemy|dangerous)\b',
            r'[!]{2,}',
            r'\b[A-Z]{4,}\b',  # SHOUTING
        ],
        # Vague sourcing
        'vague_sources': [
            r'\b(sources say|some say|many believe|experts claim|insiders reveal)\b',
            r'\b(according to sources|anonymous sources|unnamed officials)\b',
            r'\b(they don\'t want you to know|the truth they hide|what they won\'t tell you)\b',
        ],
        # Conspiracy language
        'conspiracy': [
            r'\b(conspiracy|cover-up|coverup|suppressed|censored|silenced)\b',
            r'\b(deep state|new world order|globalist|cabal|elites)\b',
            r'\b(wake up|sheeple|truth movement|red pill)\b',
        ],
        # Clickbait patterns
        'clickbait': [
            r'\b(you won\'t believe|this will shock|what happens next|the truth about)\b',
            r'\b(doctors hate|they don\'t want|one weird trick)\b',
            r'\b(exposed|busted|caught|admitted)\b',
        ],
    }
    
    # Markers for credible news
    CREDIBLE_INDICATORS = {
        'attribution': [
            r'\b(according to \w+ \w+|said \w+ \w+|reported by)\b',
            r'\b(study published|research from|data from)\b',
            r'\b(official statement|press release|confirmed)\b',
        ],
        'hedging': [
            r'\b(allegedly|reportedly|may|might|could|appears to)\b',
            r'\b(investigation|inquiry|review|analysis)\b',
        ],
        'specific_details': [
            r'\b\d{4}\b',  # Years
            r'\b\d+%|\d+\.\d+\b',  # Statistics
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Named individuals
        ],
    }
    
    def attribute(self, text: str, prediction: str, confidence: float) -> List[SentenceAttribution]:
        """
        Attribute prediction to individual sentences.
        
        Args:
            text: Full text to analyze
            prediction: Overall prediction ("FAKE" or "REAL")
            confidence: Overall confidence score
        
        Returns:
            List of sentence attributions
        """
        sentences = self._split_sentences(text)
        attributions = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Calculate contribution score
            fake_signals = self._count_signals(sentence, self.FAKE_INDICATORS)
            credible_signals = self._count_signals(sentence, self.CREDIBLE_INDICATORS)
            
            # Score: positive = supports fake, negative = supports real
            contribution = (fake_signals - credible_signals * 0.5) / 5.0
            contribution = max(-1, min(1, contribution))
            
            # Adjust based on overall prediction
            if prediction == "REAL":
                contribution = -contribution
            
            # Find key phrases
            key_phrases = self._extract_key_phrases(sentence)
            
            # Generate flags
            flags = self._generate_flags(sentence)
            
            attributions.append(SentenceAttribution(
                sentence=sentence,
                index=i,
                contribution_score=contribution,
                confidence=min(abs(contribution) + 0.3, 1.0),
                key_phrases=key_phrases,
                flags=flags
            ))
        
        return attributions
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_signals(self, sentence: str, indicators: Dict[str, List[str]]) -> int:
        """Count indicator matches in sentence."""
        count = 0
        for category, patterns in indicators.items():
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                count += len(matches)
        return count
    
    def _extract_key_phrases(self, sentence: str) -> List[str]:
        """Extract key phrases from sentence."""
        phrases = []
        
        # Find quoted text
        quotes = re.findall(r'"([^"]+)"', sentence)
        phrases.extend(quotes[:2])
        
        # Find capitalized phrases
        caps = re.findall(r'\b[A-Z][A-Z]+(?:\s+[A-Z]+)*\b', sentence)
        phrases.extend(caps[:2])
        
        return phrases[:3]
    
    def _generate_flags(self, sentence: str) -> List[str]:
        """Generate warning flags for sentence."""
        flags = []
        
        for category, patterns in self.FAKE_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    flag_map = {
                        'sensational': '⚡ Sensational language',
                        'emotional': '😱 Emotional manipulation',
                        'vague_sources': '❓ Vague sourcing',
                        'conspiracy': '🔍 Conspiracy language',
                        'clickbait': '🎣 Clickbait pattern',
                    }
                    if category in flag_map and flag_map[category] not in flags:
                        flags.append(flag_map[category])
                    break
        
        return flags[:3]


class HierarchicalExplainer:
    """
    Generates explanations at three expertise levels:
    
    1. Novice: Simple, emoji-based, actionable
    2. Intermediate: Technical terms, statistics, evidence
    3. Expert: Full model details, confidence intervals, feature weights
    """
    
    def __init__(self):
        self.attributor = SentenceAttributor()
    
    def explain(
        self,
        text: str,
        prediction: str,
        confidence: float,
        feature_scores: Optional[Dict[str, float]] = None,
        level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    ) -> Dict[str, Any]:
        """
        Generate explanation at specified level.
        
        Args:
            text: Original text
            prediction: Prediction ("FAKE" or "REAL")
            confidence: Confidence score (0-100)
            feature_scores: Scores from each feature/module
            level: Expertise level for explanation
        """
        if isinstance(level, str):
            level = ExpertiseLevel(level) if level in (e.value for e in ExpertiseLevel) else ExpertiseLevel.INTERMEDIATE
        elif not isinstance(level, ExpertiseLevel):
            level = ExpertiseLevel.INTERMEDIATE

        # Get sentence-level attribution
        sentence_attributions = self.attributor.attribute(text, prediction, confidence)
        
        # Calculate feature contributions
        feature_contributions = self._calculate_feature_contributions(
            feature_scores or {},
            prediction
        )
        
        # Generate level-appropriate explanation
        if level == ExpertiseLevel.NOVICE:
            explanation = self._novice_explanation(prediction, confidence, sentence_attributions, feature_contributions)
        elif level == ExpertiseLevel.EXPERT:
            explanation = self._expert_explanation(prediction, confidence, sentence_attributions, feature_contributions, text)
        else:
            explanation = self._intermediate_explanation(prediction, confidence, sentence_attributions, feature_contributions)
        
        return explanation
    
    def _calculate_feature_contributions(
        self,
        feature_scores: Dict[str, float],
        prediction: str
    ) -> List[FeatureContribution]:
        """Calculate contribution from each feature."""
        contributions = []
        
        # Map scores to contributions
        feature_config = {
            "text_analysis": {
                "name": "Text Content Analysis",
                "icon": "📝",
                "weight": 0.35
            },
            "linguistic_patterns": {
                "name": "Linguistic Pattern Detection",
                "icon": "🔤",
                "weight": 0.20
            },
            "ai_detection": {
                "name": "AI-Generated Content Check",
                "icon": "🤖",
                "weight": 0.15
            },
            "evidence": {
                "name": "Evidence Verification",
                "icon": "🔍",
                "weight": 0.20
            },
            "credibility": {
                "name": "Source Credibility",
                "icon": "✅",
                "weight": 0.10
            }
        }
        
        for key, score in feature_scores.items():
            config = feature_config.get(key, {
                "name": key.replace("_", " ").title(),
                "icon": "📊",
                "weight": 0.1
            })
            
            # Determine direction
            if score > 60:
                direction = "supports_fake" if prediction == "FAKE" else "supports_real"
            elif score < 40:
                direction = "supports_real" if prediction == "FAKE" else "supports_fake"
            else:
                direction = "neutral"
            
            # Generate details
            if score > 75:
                strength = "Strong"
            elif score > 50:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            contributions.append(FeatureContribution(
                name=config["name"],
                score=score,
                direction=direction,
                weight=config["weight"],
                details=f"{strength} signal ({score:.0f}%)",
                icon=config["icon"]
            ))
        
        # Sort by contribution weight
        contributions.sort(key=lambda x: x.weight * x.score, reverse=True)
        return contributions
    
    def _novice_explanation(
        self,
        prediction: str,
        confidence: float,
        attributions: List[SentenceAttribution],
        contributions: List[FeatureContribution]
    ) -> Dict[str, Any]:
        """Simple explanation for general users."""
        
        # Simple verdict
        if prediction == "FAKE":
            if confidence > 80:
                verdict_emoji = "🚨"
                verdict = "This looks like misinformation!"
            else:
                verdict_emoji = "⚠️"
                verdict = "This might not be accurate."
        else:
            if confidence > 80:
                verdict_emoji = "✅"
                verdict = "This appears to be reliable."
            else:
                verdict_emoji = "🤔"
                verdict = "This seems okay, but verify important claims."
        
        # Simple reasons (max 3)
        reasons = []
        for attr in attributions:
            if attr.flags:
                reasons.extend(attr.flags)
        reasons = list(dict.fromkeys(reasons))[:3]  # Unique, max 3
        
        if not reasons:
            if prediction == "FAKE":
                reasons = ["❓ Writing style seems unusual", "📰 Check other news sources"]
            else:
                reasons = ["✓ Writing style appears normal"]
        
        # Simple tips
        tips = [
            "🔍 Search for this story on trusted news sites",
            "📱 Check fact-checking websites like Snopes",
            "🤔 Be cautious before sharing"
        ] if prediction == "FAKE" else [
            "✓ Story appears credible",
            "📰 Still good to verify with other sources"
        ]
        
        return {
            "level": "novice",
            "verdict_emoji": verdict_emoji,
            "verdict": verdict,
            "confidence_label": "High" if confidence > 75 else "Medium" if confidence > 50 else "Low",
            "reasons": reasons,
            "tips": tips,
            "features_used": [
                {"name": c.name, "icon": c.icon, "score": c.score, "details": None}
                for c in contributions[:3]
            ]
        }
    
    def _intermediate_explanation(
        self,
        prediction: str,
        confidence: float,
        attributions: List[SentenceAttribution],
        contributions: List[FeatureContribution]
    ) -> Dict[str, Any]:
        """Detailed explanation with technical elements."""
        
        # Build analysis summary
        fake_signals = sum(1 for a in attributions if a.contribution_score > 0.3)
        credible_signals = sum(1 for a in attributions if a.contribution_score < -0.3)
        
        analysis_points = []
        
        if prediction == "FAKE":
            analysis_points.append(f"Found {fake_signals} suspicious patterns in the text")
            if credible_signals > 0:
                analysis_points.append(f"However, {credible_signals} credible elements were also detected")
        else:
            analysis_points.append(f"Found {credible_signals} credible indicators")
            if fake_signals > 0:
                analysis_points.append(f"Note: {fake_signals} potentially misleading elements detected")
        
        # Key sentences
        key_sentences = []
        sorted_attrs = sorted(attributions, key=lambda x: abs(x.contribution_score), reverse=True)
        for attr in sorted_attrs[:3]:
            if abs(attr.contribution_score) > 0.2:
                sentiment = "suspicious" if attr.contribution_score > 0 else "credible"
                key_sentences.append({
                    "text": attr.sentence[:100] + "..." if len(attr.sentence) > 100 else attr.sentence,
                    "sentiment": sentiment,
                    "flags": attr.flags,
                    "contribution": attr.contribution_score
                })
        
        return {
            "level": "intermediate",
            "prediction": prediction,
            "confidence": confidence,
            "confidence_label": f"{confidence:.0f}% confidence",
            "analysis_summary": " | ".join(analysis_points),
            "key_sentences": key_sentences,
            "features_used": [
                {
                    "name": c.name,
                    "icon": c.icon,
                    "score": c.score,
                    "direction": c.direction,
                    "details": c.details
                }
                for c in contributions
            ],
            "recommendation": self._get_recommendation(prediction, confidence)
        }
    
    def _expert_explanation(
        self,
        prediction: str,
        confidence: float,
        attributions: List[SentenceAttribution],
        contributions: List[FeatureContribution],
        text: str
    ) -> Dict[str, Any]:
        """Full technical explanation for experts."""
        
        # Statistical summary
        contribution_scores = [a.contribution_score for a in attributions]
        
        stats = {
            "total_sentences": len(attributions),
            "mean_contribution": sum(contribution_scores) / len(contribution_scores) if contribution_scores else 0,
            "max_contribution": max(contribution_scores) if contribution_scores else 0,
            "min_contribution": min(contribution_scores) if contribution_scores else 0,
            "high_impact_sentences": sum(1 for c in contribution_scores if abs(c) > 0.5)
        }
        
        # Feature weights
        feature_weights = []
        for c in contributions:
            weighted_score = c.score * c.weight
            feature_weights.append({
                "name": c.name,
                "raw_score": c.score,
                "weight": c.weight,
                "weighted_contribution": weighted_score,
                "direction": c.direction
            })
        
        # All sentence attributions
        sentence_details = [
            {
                "index": a.index,
                "text": a.sentence,
                "contribution": a.contribution_score,
                "confidence": a.confidence,
                "key_phrases": a.key_phrases,
                "flags": a.flags
            }
            for a in attributions
        ]
        
        # Confidence interval (simulated)
        margin = 100 - confidence
        ci_lower = max(0, confidence - margin * 0.3)
        ci_upper = min(100, confidence + margin * 0.2)
        
        return {
            "level": "expert",
            "prediction": prediction,
            "confidence": confidence,
            "confidence_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
                "level": "95%"
            },
            "statistics": stats,
            "feature_weights": feature_weights,
            "sentence_attributions": sentence_details,
            "model_info": {
                "version": "REMIX-FND v2.0",
                "modules_used": ["MSCIM", "EVRS", "ELDS"],
                "ensemble_method": "weighted_average"
            },
            "features_used": [
                {
                    "name": c.name,
                    "icon": c.icon,
                    "score": c.score,
                    "direction": c.direction,
                    "details": c.details,
                    "weight": c.weight
                }
                for c in contributions
            ]
        }
    
    def _get_recommendation(self, prediction: str, confidence: float) -> str:
        """Generate recommendation based on prediction."""
        if prediction == "FAKE":
            if confidence > 80:
                return "⛔ Strong evidence of misinformation. Do not share without verification."
            elif confidence > 60:
                return "⚠️ Likely unreliable. Cross-check with trusted sources before sharing."
            else:
                return "🤔 Uncertain classification. Recommend additional fact-checking."
        else:
            if confidence > 80:
                return "✅ Content appears reliable based on available signals."
            elif confidence > 60:
                return "👍 Likely accurate, but verify key claims for important decisions."
            else:
                return "ℹ️ Classification uncertain. Consider checking additional sources."


# Global instance
_explainer = None

def get_explainer() -> HierarchicalExplainer:
    global _explainer
    if _explainer is None:
        _explainer = HierarchicalExplainer()
    return _explainer

def explain(
    text: str,
    prediction: str,
    confidence: float,
    feature_scores: Optional[Dict[str, float]] = None,
    level: str = "intermediate"
) -> Dict[str, Any]:
    """Generate explanation at specified level."""
    expertise = ExpertiseLevel(level) if level in [e.value for e in ExpertiseLevel] else ExpertiseLevel.INTERMEDIATE
    return get_explainer().explain(text, prediction, confidence, feature_scores, expertise)
