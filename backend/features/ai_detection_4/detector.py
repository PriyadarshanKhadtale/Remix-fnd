"""
AI-Generated Content Detection (ELDS - Module 3)
=================================================
Explainable LLM Detection & Defense System
Implements: Multi-detector ensemble, perplexity analysis, burstiness, linguistic features
"""

import re
import math
import string
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class DetectorResult:
    """Result from a single detector."""
    name: str
    score: float  # 0-1, higher = more likely AI
    confidence: float
    details: str
    features: Dict[str, Any]


class PerplexityAnalyzer:
    """
    Analyzes text perplexity as an AI detection signal.
    
    Paper insight: AI-generated text tends to have:
    - Lower perplexity (more predictable)
    - More uniform perplexity across text
    - Smoother probability distributions
    """
    
    # Common word frequencies (simplified model)
    COMMON_WORDS = {
        'the': 0.07, 'be': 0.04, 'to': 0.03, 'of': 0.03, 'and': 0.03,
        'a': 0.02, 'in': 0.02, 'that': 0.01, 'have': 0.01, 'i': 0.01,
        'it': 0.01, 'for': 0.01, 'not': 0.01, 'on': 0.01, 'with': 0.01,
        'he': 0.01, 'as': 0.01, 'you': 0.01, 'do': 0.01, 'at': 0.01,
        'this': 0.008, 'but': 0.008, 'his': 0.008, 'by': 0.008, 'from': 0.008,
        'they': 0.007, 'we': 0.007, 'say': 0.007, 'her': 0.007, 'she': 0.007,
        'or': 0.006, 'an': 0.006, 'will': 0.006, 'my': 0.006, 'one': 0.006,
        'all': 0.005, 'would': 0.005, 'there': 0.005, 'their': 0.005
    }
    
    # Rare/unusual words that humans might use
    UNUSUAL_PATTERNS = [
        r'\b(lol|haha|hmm|ugh|omg|wtf|btw|tbh|idk|imo|fyi)\b',  # Informal
        r'\b(gonna|wanna|gotta|kinda|sorta|coulda|shoulda)\b',  # Contractions
        r'\.{3,}',  # Ellipsis
        r'!{2,}',  # Multiple exclamation
        r'\?{2,}',  # Multiple question marks
    ]
    
    def analyze(self, text: str) -> DetectorResult:
        """Analyze perplexity characteristics."""
        words = text.lower().split()
        if len(words) < 5:
            return DetectorResult(
                name="Perplexity Analysis",
                score=0.5,
                confidence=0.3,
                details="Text too short for reliable analysis",
                features={}
            )
        
        # Calculate pseudo-perplexity based on word frequency
        log_probs = []
        for word in words:
            word_clean = word.strip(string.punctuation)
            if word_clean:
                # Use frequency or assign low probability for rare words
                prob = self.COMMON_WORDS.get(word_clean, 0.0001)
                log_probs.append(math.log(prob))
        
        if not log_probs:
            return DetectorResult(
                name="Perplexity Analysis",
                score=0.5,
                confidence=0.3,
                details="Could not analyze text",
                features={}
            )
        
        # Calculate perplexity metrics
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-avg_log_prob)
        
        # Variance in log probabilities (AI tends to be more uniform)
        variance = np.var(log_probs) if len(log_probs) > 1 else 0
        
        # Check for human-like irregularities
        text_lower = text.lower()
        unusual_count = sum(1 for pattern in self.UNUSUAL_PATTERNS 
                          if re.search(pattern, text_lower, re.IGNORECASE))
        
        # Scoring: Lower perplexity + lower variance = more AI-like
        # Normal human writing: perplexity ~50-200, variance ~2-5
        # AI writing: perplexity ~20-80, variance ~0.5-2
        
        perplexity_score = 1.0 - min(perplexity / 200, 1.0)  # Lower = more AI
        variance_score = 1.0 - min(variance / 5, 1.0)  # Lower = more AI
        unusual_penalty = min(unusual_count * 0.15, 0.3)  # Unusual = more human
        
        ai_score = (perplexity_score * 0.4 + variance_score * 0.4) * (1 - unusual_penalty)
        ai_score = max(0, min(ai_score, 1))
        
        return DetectorResult(
            name="Perplexity Analysis",
            score=ai_score,
            confidence=0.75,
            details=f"Perplexity: {perplexity:.1f}, Variance: {variance:.2f}, Unusual patterns: {unusual_count}",
            features={
                "perplexity": perplexity,
                "log_prob_variance": variance,
                "unusual_patterns": unusual_count,
                "common_word_ratio": len([w for w in words if w.strip(string.punctuation).lower() in self.COMMON_WORDS]) / len(words)
            }
        )


class BurstinessAnalyzer:
    """
    Analyzes burstiness - the tendency for words to appear in clusters.
    
    Paper insight: Human writing shows "burstiness" (words cluster together),
    while AI-generated text has more uniform word distribution.
    """
    
    def analyze(self, text: str) -> DetectorResult:
        """Analyze word burstiness patterns."""
        words = [w.lower().strip(string.punctuation) for w in text.split() if w.strip(string.punctuation)]
        
        if len(words) < 20:
            return DetectorResult(
                name="Burstiness Analysis",
                score=0.5,
                confidence=0.3,
                details="Text too short for burstiness analysis",
                features={}
            )
        
        # Calculate word positions
        word_positions = defaultdict(list)
        for i, word in enumerate(words):
            if len(word) > 3:  # Only content words
                word_positions[word].append(i)
        
        # Calculate burstiness for repeated words
        burstiness_scores = []
        for word, positions in word_positions.items():
            if len(positions) >= 2:
                # Calculate gaps between occurrences
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps:
                    mean_gap = np.mean(gaps)
                    std_gap = np.std(gaps)
                    # Burstiness: std/mean ratio (higher = more bursty = more human)
                    if mean_gap > 0:
                        burstiness = std_gap / mean_gap
                        burstiness_scores.append(burstiness)
        
        if not burstiness_scores:
            return DetectorResult(
                name="Burstiness Analysis",
                score=0.5,
                confidence=0.4,
                details="Not enough repeated words for analysis",
                features={}
            )
        
        avg_burstiness = np.mean(burstiness_scores)
        
        # Human text: burstiness ~0.5-1.5
        # AI text: burstiness ~0.1-0.5 (more uniform)
        
        # Lower burstiness = more AI-like
        ai_score = 1.0 - min(avg_burstiness / 1.5, 1.0)
        ai_score = max(0, min(ai_score, 1))
        
        return DetectorResult(
            name="Burstiness Analysis",
            score=ai_score,
            confidence=0.7,
            details=f"Average burstiness: {avg_burstiness:.3f} ({'uniform/AI-like' if avg_burstiness < 0.5 else 'bursty/human-like'})",
            features={
                "avg_burstiness": avg_burstiness,
                "words_analyzed": len(burstiness_scores)
            }
        )


class LinguisticPatternAnalyzer:
    """
    Analyzes linguistic patterns characteristic of AI vs human writing.
    """
    
    # Patterns more common in AI writing
    AI_PATTERNS = [
        (r'\b(furthermore|moreover|additionally|consequently|nevertheless)\b', 'transition_words', 0.1),
        (r'\b(it is important to note|it should be noted|it is worth mentioning)\b', 'hedging_phrases', 0.15),
        (r'\b(in conclusion|to summarize|in summary|overall)\b', 'summary_markers', 0.1),
        (r'\b(firstly|secondly|thirdly|finally)\b', 'enumeration', 0.08),
        (r'\b(delve|leverage|utilize|facilitate|implement)\b', 'corporate_speak', 0.1),
        (r'\b(comprehensive|robust|innovative|dynamic|streamlined)\b', 'buzzwords', 0.08),
        (r'\b(I cannot|I don\'t have|as an AI|I\'m an AI)\b', 'ai_disclosure', 0.3),
        (r'\b(Here are|Here\'s a|Let me|I\'d be happy to)\b', 'assistant_phrases', 0.2),
    ]
    
    # Patterns more common in human writing
    HUMAN_PATTERNS = [
        (r'\b(um|uh|like|you know|basically|literally)\b', 'filler_words', -0.1),
        (r'[!?]{2,}', 'emphatic_punctuation', -0.1),
        (r'\b(lmao|rofl|smh|ngl|istg)\b', 'internet_slang', -0.15),
        (r'[A-Z]{3,}', 'shouting_caps', -0.1),
        (r'\.{3,}', 'trailing_off', -0.08),
        (r'\b(I think|I feel|I believe|in my opinion|imo)\b', 'personal_opinion', -0.05),
        (r'[😀-🙏🌀-🗿]', 'emoji', -0.1),
    ]
    
    def analyze(self, text: str) -> DetectorResult:
        """Analyze linguistic patterns."""
        ai_score_adjustment = 0.5  # Start neutral
        matches = []
        
        for pattern, name, weight in self.AI_PATTERNS:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                ai_score_adjustment += weight * min(count, 3)
                matches.append(f"AI-like: {name} ({count}x)")
        
        for pattern, name, weight in self.HUMAN_PATTERNS:
            count = len(re.findall(pattern, text, re.IGNORECASE))
            if count > 0:
                ai_score_adjustment += weight * min(count, 3)  # weight is negative
                matches.append(f"Human-like: {name} ({count}x)")
        
        # Sentence structure analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # Sentence length variance
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(lengths) if len(lengths) > 1 else 0
            
            # AI tends to have uniform sentence lengths
            if length_variance < 20:  # Low variance
                ai_score_adjustment += 0.1
                matches.append("Uniform sentence lengths")
            elif length_variance > 50:  # High variance
                ai_score_adjustment -= 0.1
                matches.append("Varied sentence lengths")
        
        ai_score = max(0, min(ai_score_adjustment, 1))
        
        return DetectorResult(
            name="Linguistic Patterns",
            score=ai_score,
            confidence=0.65,
            details="; ".join(matches[:5]) if matches else "No distinctive patterns found",
            features={
                "patterns_found": matches,
                "raw_score": ai_score_adjustment
            }
        )


class RepetitionAnalyzer:
    """
    Analyzes repetition patterns.
    AI tends to repeat phrases and structures more than humans.
    """
    
    def analyze(self, text: str) -> DetectorResult:
        """Analyze repetition in text."""
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(words) < 20 or len(sentences) < 2:
            return DetectorResult(
                name="Repetition Analysis",
                score=0.5,
                confidence=0.3,
                details="Text too short for repetition analysis",
                features={}
            )
        
        # 1. N-gram repetition
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
        repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
        
        # 2. Sentence start patterns
        sentence_starts = [s.split()[:3] if len(s.split()) >= 3 else s.split() for s in sentences]
        start_patterns = [' '.join(s).lower() for s in sentence_starts]
        start_repetition = len(start_patterns) - len(set(start_patterns))
        
        # Calculate repetition score
        bigram_ratio = repeated_bigrams / max(len(bigrams), 1)
        trigram_ratio = repeated_trigrams / max(len(trigrams), 1)
        start_ratio = start_repetition / max(len(sentences), 1)
        
        # Higher repetition = more AI-like
        repetition_score = (bigram_ratio * 0.3 + trigram_ratio * 0.4 + start_ratio * 0.3)
        ai_score = min(repetition_score * 5, 1.0)  # Scale up
        
        return DetectorResult(
            name="Repetition Analysis",
            score=ai_score,
            confidence=0.6,
            details=f"Repeated bigrams: {repeated_bigrams}, trigrams: {repeated_trigrams}, sentence starts: {start_repetition}",
            features={
                "repeated_bigrams": repeated_bigrams,
                "repeated_trigrams": repeated_trigrams,
                "repeated_starts": start_repetition
            }
        )


class VocabularyRichnessAnalyzer:
    """
    Analyzes vocabulary richness and diversity.
    """
    
    def analyze(self, text: str) -> DetectorResult:
        """Analyze vocabulary richness."""
        words = [w.lower().strip(string.punctuation) for w in text.split() 
                if w.strip(string.punctuation)]
        
        if len(words) < 20:
            return DetectorResult(
                name="Vocabulary Richness",
                score=0.5,
                confidence=0.3,
                details="Text too short for vocabulary analysis",
                features={}
            )
        
        unique_words = set(words)
        
        # Type-Token Ratio (TTR)
        ttr = len(unique_words) / len(words)
        
        # Hapax legomena (words appearing only once)
        word_counts = Counter(words)
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = hapax / len(unique_words) if unique_words else 0
        
        # Average word length
        avg_word_length = np.mean([len(w) for w in words])
        
        # AI text tends to have:
        # - Moderate TTR (not too high, not too low)
        # - Lower hapax ratio (uses same words repeatedly)
        # - Slightly longer average word length (formal vocabulary)
        
        # Scoring
        ttr_score = abs(ttr - 0.6) * 2  # Deviation from "perfect" 0.6
        hapax_score = 1 - hapax_ratio  # Lower hapax = more AI
        length_score = (avg_word_length - 4) / 3  # Longer = more AI
        
        ai_score = (ttr_score * 0.3 + hapax_score * 0.4 + length_score * 0.3)
        ai_score = max(0, min(ai_score, 1))
        
        return DetectorResult(
            name="Vocabulary Richness",
            score=ai_score,
            confidence=0.55,
            details=f"TTR: {ttr:.3f}, Hapax ratio: {hapax_ratio:.3f}, Avg word length: {avg_word_length:.1f}",
            features={
                "ttr": ttr,
                "hapax_ratio": hapax_ratio,
                "avg_word_length": avg_word_length,
                "unique_words": len(unique_words),
                "total_words": len(words)
            }
        )


def _char_trigrams(s: str, limit: int = 2000) -> Set[str]:
    t = re.sub(r"\s+", " ", s.lower().strip())[:limit]
    if len(t) < 3:
        return set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


class HC3CorpusSimilarityDetector:
    """
    Retrieval-style signal (Table 2): similarity to known ChatGPT answers from HC3 (sample).
    Lazy-loads references to keep import/start fast.
    """

    def __init__(self, max_refs: int = 100, max_lines: int = 400):
        self.max_refs = max_refs
        self.max_lines = max_lines
        self._loaded = False
        self._ref_trigrams: List[Set[str]] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from features.ai_detection_4.load_hc3_dataset import load_hc3_dataset
        except ImportError:
            return
        _, ai_samples = load_hc3_dataset(max_samples=self.max_lines)
        for row in ai_samples:
            txt = (row.get("text") or "").strip()
            if len(txt) < 80:
                continue
            self._ref_trigrams.append(_char_trigrams(txt[:1500]))
            if len(self._ref_trigrams) >= self.max_refs:
                break

    def analyze(self, text: str) -> DetectorResult:
        self._ensure_loaded()
        if len(text.strip()) < 30:
            return DetectorResult(
                name="Corpus retrieval (HC3)",
                score=0.5,
                confidence=0.25,
                details="Text too short for corpus similarity",
                features={"refs": len(self._ref_trigrams)},
            )
        if not self._ref_trigrams:
            return DetectorResult(
                name="Corpus retrieval (HC3)",
                score=0.5,
                confidence=0.2,
                details="HC3 reference corpus not available",
                features={"refs": 0},
            )

        q = _char_trigrams(text)
        if not q:
            return DetectorResult(
                name="Corpus retrieval (HC3)",
                score=0.5,
                confidence=0.3,
                details="Could not extract trigrams",
                features={},
            )

        best = 0.0
        for ref in self._ref_trigrams:
            inter = len(q & ref)
            if inter == 0:
                continue
            union = len(q | ref) or 1
            best = max(best, inter / union)

        # Map Jaccard to AI-likeness (empirical cap ~0.15+ for strong overlap)
        ai_score = float(max(0.0, min(1.0, best / 0.12)))
        return DetectorResult(
            name="Corpus retrieval (HC3)",
            score=ai_score,
            confidence=0.6,
            details=f"Max trigram Jaccard vs HC3 AI answers: {best:.4f} ({len(self._ref_trigrams)} refs)",
            features={"max_jaccard": best, "refs": len(self._ref_trigrams)},
        )


class AIContentDetector:
    """
    Multi-detector ensemble for AI-generated content detection.
    
    Paper Table 2 lists six parallel detectors; this stack uses six complementary
    lightweight signals including HC3 retrieval similarity (Table 2 “Retrieval”).
    """
    
    def __init__(self):
        # Weights sum to 1.0; sixth slot = corpus retrieval vs HC3 ChatGPT answers.
        self.detectors = [
            (PerplexityAnalyzer(), 0.22),
            (BurstinessAnalyzer(), 0.18),
            (LinguisticPatternAnalyzer(), 0.18),
            (RepetitionAnalyzer(), 0.14),
            (VocabularyRichnessAnalyzer(), 0.13),
            (HC3CorpusSimilarityDetector(), 0.15),
        ]
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Run all detectors and aggregate results.
        
        Returns:
            Dictionary with detection results, confidence, and explanation
        """
        if not text or len(text.strip()) < 10:
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "probability": 0.5,
                "verdict": "Text too short for analysis",
                "detectors": [],
                "explanation": "Cannot analyze text shorter than 10 characters."
            }
        
        # Run all detectors
        results = []
        for detector, weight in self.detectors:
            try:
                result = detector.analyze(text)
                results.append((result, weight))
            except Exception as e:
                print(f"Detector {detector.__class__.__name__} failed: {e}")
                continue
        
        if not results:
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "probability": 0.5,
                "verdict": "Analysis failed",
                "detectors": [],
                "explanation": "All detectors failed to analyze the text."
            }
        
        # Weighted ensemble
        total_weight = sum(w for _, w in results)
        weighted_score = sum(r.score * w for r, w in results) / total_weight
        avg_confidence = sum(r.confidence * w for r, w in results) / total_weight
        
        # Determine verdict
        is_ai = weighted_score > 0.55
        
        if weighted_score > 0.75:
            verdict = "Highly likely AI-generated"
        elif weighted_score > 0.55:
            verdict = "Likely AI-generated"
        elif weighted_score > 0.45:
            verdict = "Uncertain - could be either"
        elif weighted_score > 0.25:
            verdict = "Likely human-written"
        else:
            verdict = "Highly likely human-written"
        
        # Build explanation
        explanations = []
        for result, weight in sorted(results, key=lambda x: x[0].score, reverse=True):
            indicator = "🤖" if result.score > 0.5 else "👤"
            explanations.append(f"{indicator} {result.name}: {result.details}")
        
        return {
            "is_ai_generated": is_ai,
            "confidence": avg_confidence * 100,
            "probability": weighted_score * 100,
            "verdict": verdict,
            "detectors": [
                {
                    "name": r.name,
                    "score": r.score * 100,
                    "confidence": r.confidence * 100,
                    "details": r.details,
                    "features": r.features,
                    "weight": w
                }
                for r, w in results
            ],
            "explanation": "\n".join(explanations),
            "summary": self._generate_summary(results, is_ai)
        }
    
    def _generate_summary(self, results: List[Tuple[DetectorResult, float]], is_ai: bool) -> str:
        """Generate human-readable summary."""
        key_signals = []
        for result, _ in results:
            if is_ai and result.score > 0.6:
                key_signals.append(f"• {result.name} indicates AI patterns")
            elif not is_ai and result.score < 0.4:
                key_signals.append(f"• {result.name} indicates human patterns")
        
        if is_ai:
            intro = "This text shows characteristics typical of AI-generated content:"
        else:
            intro = "This text shows characteristics typical of human writing:"
        
        if key_signals:
            return intro + "\n" + "\n".join(key_signals[:3])
        else:
            return "The analysis is inconclusive. Consider additional verification."


# Global instance
_detector = None

def get_detector() -> AIContentDetector:
    global _detector
    if _detector is None:
        _detector = AIContentDetector()
    return _detector

def detect(text: str) -> Dict[str, Any]:
    return get_detector().detect(text)
