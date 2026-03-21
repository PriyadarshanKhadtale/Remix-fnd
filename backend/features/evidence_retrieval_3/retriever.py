"""
Evidence Retriever (EVRS - Module 2)
====================================
Evidence-Based Verification & Retrieval System
Implements: FAISS vector search, expanded knowledge base, uncertainty-based depth control
"""

import os
import re
import json
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers not installed. Using keyword search only.")

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Using keyword search only.")


@dataclass
class EvidenceItem:
    """A piece of evidence."""
    source: str
    title: str
    snippet: str
    relevance_score: float
    supports_claim: Optional[bool]
    url: Optional[str] = None
    category: Optional[str] = None
    stance: Optional[str] = None  # supports, refutes, neutral


class ExpandedKnowledgeBase:
    """
    Expanded knowledge base with LIAR dataset (12.8K political claims) + hand-crafted facts.
    Supports both keyword search and FAISS vector search.
    
    Data sources:
    - LIAR Dataset: 12,836 fact-checked political claims from PolitiFact
    - Hand-crafted: 30 curated facts on health, science, conspiracies
    """
    
    # Hand-crafted facts as fallback/supplement
    HANDCRAFTED_FACTS = [
        # === HEALTH & MEDICINE ===
        {
            "category": "health",
            "title": "COVID-19 vaccines are safe and effective",
            "content": "Major health organizations including WHO, CDC, FDA, and EMA have confirmed that approved COVID-19 vaccines (Pfizer, Moderna, AstraZeneca, J&J) are safe and effective at preventing severe illness, hospitalization, and death. Clinical trials involved over 100,000 participants.",
            "source": "World Health Organization",
            "stance": "established_fact",
            "keywords": ["covid", "vaccine", "vaccines", "coronavirus", "safe", "effective", "pfizer", "moderna"]
        },
        {
            "category": "health",
            "title": "5G technology does not spread viruses",
            "content": "Radio waves, including those used by 5G networks, cannot spread viruses. Viruses are biological pathogens that spread through respiratory droplets, contact with infected surfaces, or bodily fluids. 5G uses non-ionizing radiation that cannot alter DNA or spread biological agents.",
            "source": "IEEE & WHO",
            "stance": "established_fact",
            "keywords": ["5g", "virus", "spread", "coronavirus", "radiation", "waves", "technology"]
        },
        {
            "category": "health",
            "title": "There is no miracle cure for cancer",
            "content": "Cancer treatment requires proper medical care including surgery, chemotherapy, radiation, immunotherapy, or targeted therapy. No single food, supplement, essential oil, or alternative treatment has been scientifically proven to cure cancer on its own. Claims of miracle cures are typically fraudulent.",
            "source": "American Cancer Society & NCI",
            "stance": "established_fact",
            "keywords": ["cancer", "cure", "miracle", "treatment", "natural", "remedy", "alternative"]
        },
        {
            "category": "health",
            "title": "Vaccines do not cause autism",
            "content": "Extensive scientific research involving millions of children has conclusively shown no link between vaccines and autism. The original 1998 study claiming a link was retracted and its author lost his medical license for ethical violations and data manipulation.",
            "source": "CDC, WHO, Lancet Retraction",
            "stance": "established_fact",
            "keywords": ["vaccine", "autism", "mmr", "children", "cause", "link"]
        },
        {
            "category": "health",
            "title": "Hydroxychloroquine is not proven effective for COVID-19",
            "content": "Multiple large-scale clinical trials (RECOVERY, SOLIDARITY) found hydroxychloroquine does not provide benefit for COVID-19 patients and may cause harmful side effects. FDA revoked emergency use authorization in June 2020.",
            "source": "FDA, NIH, WHO",
            "stance": "established_fact",
            "keywords": ["hydroxychloroquine", "covid", "treatment", "malaria", "cure"]
        },
        {
            "category": "health",
            "title": "Ivermectin is not approved for COVID-19 treatment",
            "content": "While ivermectin is an approved antiparasitic drug, clinical trials have not shown it to be effective against COVID-19 in humans at safe doses. FDA and WHO advise against using ivermectin for COVID-19 outside of clinical trials.",
            "source": "FDA, WHO, Merck",
            "stance": "established_fact",
            "keywords": ["ivermectin", "covid", "treatment", "horse", "dewormer", "parasite"]
        },
        {
            "category": "health",
            "title": "Masks help reduce COVID-19 transmission",
            "content": "Scientific studies demonstrate that properly worn masks reduce transmission of respiratory viruses including SARS-CoV-2. N95/KN95 masks provide highest protection, followed by surgical masks, then cloth masks.",
            "source": "CDC, Lancet Studies",
            "stance": "established_fact",
            "keywords": ["mask", "masks", "covid", "transmission", "protection", "n95"]
        },
        
        # === CLIMATE & ENVIRONMENT ===
        {
            "category": "climate",
            "title": "Climate change is real and human-caused",
            "content": "97% of climate scientists agree that climate change is real and primarily caused by human activities, especially burning fossil fuels. Global temperatures have risen 1.1°C since pre-industrial times, with observable effects on weather patterns, ice sheets, and sea levels.",
            "source": "NASA, IPCC, NOAA",
            "stance": "scientific_consensus",
            "keywords": ["climate", "change", "global", "warming", "science", "scientists", "consensus", "hoax"]
        },
        {
            "category": "climate",
            "title": "Arctic ice is declining significantly",
            "content": "Arctic sea ice extent has declined by approximately 13% per decade since satellite records began in 1979. The Arctic is warming 2-3 times faster than the global average, a phenomenon called Arctic amplification.",
            "source": "NASA, NSIDC",
            "stance": "established_fact",
            "keywords": ["arctic", "ice", "melting", "polar", "sea", "decline"]
        },
        {
            "category": "climate",
            "title": "Sea levels are rising",
            "content": "Global mean sea level has risen about 8-9 inches (21-24 cm) since 1880. The rate of rise is accelerating, with projections of 1-4 feet additional rise by 2100 depending on emissions scenarios.",
            "source": "NOAA, IPCC",
            "stance": "established_fact",
            "keywords": ["sea", "level", "rise", "rising", "ocean", "coast", "flooding"]
        },
        
        # === SCIENCE & SPACE ===
        {
            "category": "science",
            "title": "Earth is approximately spherical (oblate spheroid)",
            "content": "Earth's spherical shape has been proven through centuries of scientific observation, satellite imagery, circumnavigation, and space exploration. Earth is slightly flattened at the poles (oblate spheroid) due to rotation.",
            "source": "NASA, ESA, All Space Agencies",
            "stance": "established_fact",
            "keywords": ["flat", "earth", "round", "globe", "sphere", "shape", "planet"]
        },
        {
            "category": "science",
            "title": "Moon landings were real",
            "content": "The Apollo moon landings (1969-1972) are documented historical events verified by independent sources worldwide, including the Soviet Union during the Cold War. Physical evidence includes 842 pounds of moon rocks, retroreflectors still used today, and photos from lunar orbiters.",
            "source": "NASA, International Verification",
            "stance": "established_fact",
            "keywords": ["moon", "landing", "fake", "apollo", "nasa", "hoax", "astronaut"]
        },
        {
            "category": "science",
            "title": "Evolution is supported by extensive evidence",
            "content": "Biological evolution is supported by evidence from fossils, genetics, comparative anatomy, observed speciation, and molecular biology. It is the foundational theory of modern biology accepted by virtually all scientists in relevant fields.",
            "source": "National Academy of Sciences",
            "stance": "scientific_consensus",
            "keywords": ["evolution", "darwin", "species", "natural", "selection", "theory"]
        },
        
        # === POLITICS & ELECTIONS ===
        {
            "category": "politics",
            "title": "2020 US Presidential Election was secure",
            "content": "The 2020 US election was described as 'the most secure in American history' by CISA. Over 60 court cases challenging results were dismissed for lack of evidence. State audits, including Republican-led ones, confirmed results.",
            "source": "CISA, State Election Officials, Courts",
            "stance": "verified_claim",
            "keywords": ["election", "2020", "fraud", "stolen", "vote", "biden", "trump", "rigged"]
        },
        {
            "category": "politics",
            "title": "Fact-checking organizations use documented evidence",
            "content": "Reputable fact-checking organizations (Snopes, PolitiFact, FactCheck.org, AP Fact Check) use documented evidence, primary sources, and expert consultation. They are certified by the International Fact-Checking Network (IFCN).",
            "source": "IFCN, Poynter Institute",
            "stance": "established_fact",
            "keywords": ["fact", "check", "verify", "political", "claims", "snopes", "politifact"]
        },
        
        # === TECHNOLOGY & AI ===
        {
            "category": "technology",
            "title": "AI-generated content can be misleading",
            "content": "AI systems like ChatGPT, GPT-4, and image generators can create realistic but false text, images, and videos (deepfakes). Always verify AI-generated information with reliable primary sources.",
            "source": "AI Ethics Guidelines",
            "stance": "established_fact",
            "keywords": ["ai", "artificial", "intelligence", "chatgpt", "generated", "fake", "deepfake"]
        },
        {
            "category": "technology",
            "title": "Social media algorithms can create filter bubbles",
            "content": "Social media recommendation algorithms tend to show users content similar to what they've engaged with before, potentially creating 'filter bubbles' or 'echo chambers' that reinforce existing beliefs.",
            "source": "MIT Media Lab Research",
            "stance": "research_finding",
            "keywords": ["social", "media", "algorithm", "bubble", "echo", "chamber", "facebook", "twitter"]
        },
        {
            "category": "technology",
            "title": "Quantum computers cannot yet break encryption",
            "content": "While theoretically possible, current quantum computers lack sufficient qubits and error correction to break modern encryption. Practical quantum threats to encryption are estimated to be 10-20+ years away.",
            "source": "NIST, IBM Research",
            "stance": "current_status",
            "keywords": ["quantum", "computer", "encryption", "security", "break", "hack"]
        },
        
        # === ECONOMICS & FINANCE ===
        {
            "category": "economics",
            "title": "Cryptocurrency values are highly volatile",
            "content": "Cryptocurrency prices can fluctuate dramatically - Bitcoin has seen drops of 50%+ multiple times. Crypto investments carry significant risk and are not guaranteed to increase in value.",
            "source": "SEC, Financial Regulators",
            "stance": "established_fact",
            "keywords": ["bitcoin", "crypto", "cryptocurrency", "investment", "guaranteed", "returns"]
        },
        {
            "category": "economics",
            "title": "Get-rich-quick schemes are typically scams",
            "content": "Legitimate investments rarely promise guaranteed high returns with no risk. Schemes promising quick wealth through minimal effort are often Ponzi schemes, pyramid schemes, or fraud.",
            "source": "FTC, SEC",
            "stance": "consumer_protection",
            "keywords": ["rich", "quick", "money", "scheme", "investment", "guaranteed", "passive", "income"]
        },
        
        # === FOOD & NUTRITION ===
        {
            "category": "nutrition",
            "title": "GMO foods are safe to eat",
            "content": "Genetically modified foods approved for sale have been evaluated for safety by regulatory agencies worldwide (FDA, EFSA, WHO). No credible scientific evidence shows GMO foods are harmful to human health.",
            "source": "WHO, FDA, National Academies",
            "stance": "scientific_consensus",
            "keywords": ["gmo", "genetically", "modified", "food", "safe", "dangerous", "organic"]
        },
        {
            "category": "nutrition",
            "title": "Detox diets have no scientific basis",
            "content": "The human body naturally detoxifies through the liver, kidneys, and other organs. Commercial 'detox' products and extreme detox diets have no proven health benefits and may be harmful.",
            "source": "Harvard Health, NHS",
            "stance": "established_fact",
            "keywords": ["detox", "cleanse", "toxins", "diet", "juice", "liver"]
        },
        
        # === CONSPIRACY THEORIES (DEBUNKED) ===
        {
            "category": "conspiracy",
            "title": "Chemtrails are not real",
            "content": "Contrails (condensation trails) from aircraft are water vapor that freezes in cold upper atmosphere. There is no evidence of secret chemical spraying programs. Atmospheric scientists have repeatedly debunked 'chemtrail' claims.",
            "source": "EPA, Atmospheric Scientists",
            "stance": "debunked",
            "keywords": ["chemtrail", "chemtrails", "spray", "airplane", "contrail", "chemicals", "sky"]
        },
        {
            "category": "conspiracy",
            "title": "There is no evidence of a 'New World Order' conspiracy",
            "content": "Claims of a secret global elite controlling world events lack credible evidence. International organizations like the UN operate through documented, transparent processes with member state participation.",
            "source": "Academic Research",
            "stance": "debunked",
            "keywords": ["new", "world", "order", "illuminati", "secret", "elite", "global", "control"]
        },
        {
            "category": "conspiracy",
            "title": "QAnon claims are baseless",
            "content": "QAnon conspiracy theories about secret cabals, mass arrests, and 'the storm' have been repeatedly debunked. Predicted events have consistently failed to occur. QAnon is classified as a domestic terrorism threat by FBI.",
            "source": "FBI, Academic Research",
            "stance": "debunked",
            "keywords": ["qanon", "cabal", "storm", "wwg1wga", "deep", "state", "pedophile"]
        },
        
        # === MEDIA LITERACY ===
        {
            "category": "media",
            "title": "Reliable news sources have editorial standards",
            "content": "Credible news organizations have editorial standards, fact-checking processes, corrections policies, and accountability. Look for bylines, sources cited, and separation of news from opinion.",
            "source": "Society of Professional Journalists",
            "stance": "media_literacy",
            "keywords": ["news", "reliable", "source", "media", "journalism", "fake", "real"]
        },
        {
            "category": "media",
            "title": "Satirical websites are not real news",
            "content": "Websites like The Onion, Babylon Bee, and others publish satirical content not meant to be taken as factual news. Always check if a source is satire before sharing.",
            "source": "Media Literacy Guidelines",
            "stance": "media_literacy",
            "keywords": ["satire", "onion", "babylon", "bee", "joke", "parody", "fake"]
        },
        {
            "category": "media",
            "title": "Screenshots can be easily faked",
            "content": "Screenshots of social media posts, text messages, or documents can be easily manipulated using basic editing tools. Always verify claims by finding the original source.",
            "source": "Digital Literacy Guidelines",
            "stance": "media_literacy",
            "keywords": ["screenshot", "fake", "edited", "photoshop", "manipulated", "tweet"]
        },
        
        # === HISTORY ===
        {
            "category": "history",
            "title": "The Holocaust was real and well-documented",
            "content": "The Holocaust - the systematic murder of six million Jews and millions of others by Nazi Germany - is one of the most documented events in history. Evidence includes Nazi records, survivor testimony, physical evidence, and Allied liberation documentation.",
            "source": "Yad Vashem, US Holocaust Memorial Museum",
            "stance": "established_fact",
            "keywords": ["holocaust", "nazi", "jews", "denial", "hitler", "genocide", "ww2"]
        },
        {
            "category": "history",
            "title": "9/11 was a terrorist attack, not an inside job",
            "content": "The September 11, 2001 attacks were carried out by al-Qaeda terrorists. Extensive investigations by the 9/11 Commission, NIST, and independent engineers have documented the events. 'Inside job' claims have been thoroughly debunked.",
            "source": "9/11 Commission, NIST",
            "stance": "established_fact",
            "keywords": ["911", "9/11", "inside", "job", "tower", "collapse", "terrorist"]
        },
    ]
    
    def __init__(self, use_faiss: bool = True, load_datasets: bool = True):
        self.facts = []
        self.embeddings = None
        self.index = None
        self.embedding_model = None
        self._faiss_initialized = False
        self._use_faiss = use_faiss
        
        if load_datasets:
            # Load LIAR dataset (12.8K political fact-checks from PolitiFact)
            try:
                from .load_liar_dataset import load_liar_dataset
                liar_facts = load_liar_dataset()
                self.facts.extend(liar_facts)
            except Exception as e:
                print(f"  ⚠️ LIAR dataset: {e}")
        
        # 3. Add hand-crafted facts (health, science, conspiracies)
        self.facts.extend(self.HANDCRAFTED_FACTS)
        
        print(f"  📚 Total knowledge base: {len(self.facts)} facts")
        
        # Build keyword index
        self._build_keyword_index()
        
        # FAISS index built lazily on first search
    
    def _build_keyword_index(self):
        """Build inverted index for keyword search."""
        self.keyword_index = defaultdict(list)
        for i, fact in enumerate(self.facts):
            for keyword in fact.get("keywords", []):
                self.keyword_index[keyword.lower()].append(i)
            # Also index words from title and content
            title_words = re.findall(r'\b\w+\b', fact["title"].lower())
            for word in title_words:
                if len(word) > 3:
                    self.keyword_index[word].append(i)
    
    def _build_faiss_index(self):
        """Build FAISS index for semantic search."""
        try:
            print("  Building FAISS index...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create embeddings for all facts
            texts = [f"{fact['title']}. {fact['content']}" for fact in self.facts]
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"  ✓ FAISS index built with {len(self.facts)} facts")
        except Exception as e:
            print(f"  ⚠️ Failed to build FAISS index: {e}")
            self.index = None
    
    def search(self, query: str, top_k: int = 5, use_semantic: bool = True) -> List[Dict]:
        """
        Hybrid search: combines FAISS semantic search with keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
            use_semantic: Whether to use FAISS (if available)
        """
        results = []
        
        # Initialize FAISS lazily on first search (avoids startup issues)
        if use_semantic and self._use_faiss and not self._faiss_initialized:
            if EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
                self._build_faiss_index()
            self._faiss_initialized = True
        
        # Semantic search with FAISS
        if use_semantic and self.index is not None and self.embedding_model is not None:
            query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.facts)))
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.facts) and score > 0.3:  # Threshold
                    fact = self.facts[idx]
                    results.append({
                        **fact,
                        "relevance_score": float(score),
                        "search_method": "semantic"
                    })
        
        # Keyword search (fallback or supplement)
        keyword_results = self._keyword_search(query, top_k)
        
        # Merge results
        seen_titles = {r["title"] for r in results}
        for kr in keyword_results:
            if kr["title"] not in seen_titles:
                results.append(kr)
                seen_titles.add(kr["title"])
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return results[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scores = defaultdict(float)
        
        for word in query_words:
            for fact_idx in self.keyword_index.get(word, []):
                scores[fact_idx] += 1.0
        
        # Also check content match
        for i, fact in enumerate(self.facts):
            content_lower = fact["content"].lower()
            title_lower = fact["title"].lower()
            for word in query_words:
                if len(word) > 3:
                    if word in content_lower:
                        scores[i] += 0.5
                    if word in title_lower:
                        scores[i] += 0.8
        
        # Get top results
        sorted_facts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for fact_idx, score in sorted_facts[:top_k]:
            if score > 0:
                fact = self.facts[fact_idx]
                results.append({
                    **fact,
                    "relevance_score": min(score / 5.0, 1.0),
                    "search_method": "keyword"
                })
        
        return results


class EvidenceRetriever:
    """
    Evidence-Based Verification & Retrieval System (EVRS)
    
    Features (from paper):
    - Uncertainty-based adaptive depth control
    - Hybrid retrieval (semantic + keyword)
    - Hierarchical claim verification
    - Stance classification
    """
    
    def __init__(self):
        self.kb = ExpandedKnowledgeBase()
        self.use_semantic = FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE
        self._use_neural_stance = os.environ.get("REMIX_DISABLE_NEURAL_STANCE", "").lower() not in (
            "1",
            "true",
            "yes",
        )
    
    def retrieve(
        self, 
        text: str, 
        max_results: int = 5,
        uncertainty: float = 0.5,  # From Module 1
        depth_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve evidence with uncertainty-based depth control.
        
        Args:
            text: Claim to fact-check
            max_results: Base number of results
            uncertainty: Uncertainty from Module 1 (0-1, higher = more uncertain)
            depth_override: If set (e.g. 5/10/20 from Table 1 / MC routing), use this depth directly.
        """
        # Adaptive depth based on uncertainty (paper: 5-20 documents)
        if depth_override is not None:
            adaptive_depth = max(5, min(20, int(depth_override)))
        else:
            adaptive_depth = self._calculate_depth(uncertainty, max_results)
        
        # Extract key claims
        claims = self._extract_claims(text)
        
        # Search knowledge base
        results = self.kb.search(text, top_k=adaptive_depth, use_semantic=self.use_semantic)
        
        # Build evidence items with stance classification
        evidence = []
        for r in results:
            stance = self._classify_stance(text, r)
            supports = stance == "supports"
            contradicts = stance == "refutes"
            
            evidence.append(EvidenceItem(
                source=r.get("source", "Unknown"),
                title=r.get("title", ""),
                snippet=r.get("content", "")[:300] + "..." if len(r.get("content", "")) > 300 else r.get("content", ""),
                relevance_score=r.get("relevance_score", 0),
                supports_claim=True if supports else (False if contradicts else None),
                category=r.get("category"),
                stance=stance
            ))
        
        # Generate verdict
        verdict, confidence = self._generate_verdict(text, evidence)
        
        # Helper to convert numpy types to Python types
        def to_python(val):
            if hasattr(val, 'item'):
                return val.item()
            if isinstance(val, (list, tuple)):
                return [to_python(v) for v in val]
            return val
        
        return {
            "query": text,
            "claim_keywords": claims,
            "retrieval_depth": int(adaptive_depth),
            "search_method": "hybrid (semantic + keyword)" if self.use_semantic else "keyword",
            "evidence": [
                {
                    "source": str(e.source),
                    "title": str(e.title),
                    "snippet": str(e.snippet),
                    "relevance_score": float(e.relevance_score) if e.relevance_score else 0.0,
                    "supports_claim": bool(e.supports_claim) if e.supports_claim is not None else None,
                    "category": str(e.category) if e.category else None,
                    "stance": str(e.stance) if e.stance else None
                }
                for e in evidence
            ],
            "verdict": str(verdict),
            "confidence": float(confidence),
            "evidence_summary": str(self._summarize_evidence(evidence)),
            "recommendation": str(self._get_recommendation(verdict, evidence))
        }
    
    def _calculate_depth(self, uncertainty: float, base: int) -> int:
        """
        Calculate retrieval depth based on uncertainty.
        Paper: 5-20 documents based on Module 1 uncertainty.
        """
        min_depth = 5
        max_depth = 20
        
        # Higher uncertainty = deeper search
        depth = min_depth + int((max_depth - min_depth) * uncertainty)
        return max(min_depth, min(depth, max_depth, base * 3))
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract key claims/keywords from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of',
                    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'that', 'this', 'these', 'those', 'it', 'its', 'and', 'or',
                    'but', 'if', 'not', 'no', 'yes', 'all', 'any', 'some', 'more',
                    'most', 'other', 'so', 'than', 'too', 'very', 'just', 'being'}
        
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        from collections import Counter
        counts = Counter(keywords)
        return [word for word, _ in counts.most_common(10)]
    
    def _classify_stance(self, claim: str, evidence: Dict) -> str:
        """
        Classify stance: supports, refutes, or neutral.
        Uses trained cross-encoder when models/stance_cross_encoder/best_model.pt exists.
        """
        if self._use_neural_stance:
            try:
                from .stance_encoder import get_stance_scorer
                scorer = get_stance_scorer(device="cpu")
                if scorer is not None:
                    passage = evidence.get("content") or evidence.get("title") or ""
                    label, _ = scorer.predict(claim, passage)
                    return label
            except Exception:
                pass

        claim_lower = claim.lower()
        content_lower = evidence.get("content", "").lower()
        stance_type = evidence.get("stance", "")
        
        # Check for negation patterns in claim
        negation_in_claim = any(neg in claim_lower for neg in 
            ["not", "don't", "doesn't", "isn't", "aren't", "never", "no ", "fake", "false", "hoax", "myth", "lie"])
        
        # Check if evidence debunks something
        is_debunking_evidence = stance_type in ["debunked", "established_fact"] or \
            any(term in content_lower for term in ["debunked", "no evidence", "false", "myth", "not true", "incorrect"])
        
        # Determine stance
        if is_debunking_evidence:
            if negation_in_claim:
                return "supports"  # Claim says X is false, evidence confirms X is false
            else:
                return "refutes"   # Claim says X is true, evidence says X is false
        
        # Check keyword overlap for relevance
        claim_keywords = set(re.findall(r'\b\w{4,}\b', claim_lower))
        evidence_keywords = set(evidence.get("keywords", []))
        
        if len(claim_keywords & evidence_keywords) >= 2:
            return "neutral"  # Related but stance unclear
        
        return "neutral"
    
    def _generate_verdict(self, claim: str, evidence: List[EvidenceItem]) -> Tuple[str, float]:
        """Generate verdict based on evidence."""
        if not evidence:
            return "insufficient_evidence", 0.0
        
        supports = sum(1 for e in evidence if e.supports_claim is True)
        refutes = sum(1 for e in evidence if e.supports_claim is False)
        
        total = len(evidence)
        avg_relevance = sum(e.relevance_score for e in evidence) / total
        
        if refutes > supports and refutes >= 1:
            verdict = "likely_false"
            confidence = min(avg_relevance * 100 + refutes * 10, 90)
        elif supports > refutes and supports >= 1:
            verdict = "likely_true"
            confidence = min(avg_relevance * 100 + supports * 10, 90)
        elif avg_relevance > 0.5:
            verdict = "mixed_evidence"
            confidence = avg_relevance * 60
        else:
            verdict = "insufficient_evidence"
            confidence = avg_relevance * 40
        
        return verdict, confidence
    
    def _summarize_evidence(self, evidence: List[EvidenceItem]) -> str:
        """Generate evidence summary."""
        if not evidence:
            return "No relevant evidence found."
        
        supports = sum(1 for e in evidence if e.supports_claim is True)
        refutes = sum(1 for e in evidence if e.supports_claim is False)
        neutral = sum(1 for e in evidence if e.supports_claim is None)
        
        parts = []
        if refutes > 0:
            parts.append(f"{refutes} source(s) contradict the claim")
        if supports > 0:
            parts.append(f"{supports} source(s) support the claim")
        if neutral > 0:
            parts.append(f"{neutral} source(s) provide related information")
        
        return ". ".join(parts) + "." if parts else "Evidence inconclusive."
    
    def _get_recommendation(self, verdict: str, evidence: List[EvidenceItem]) -> str:
        """Generate recommendation based on verdict."""
        recommendations = {
            "likely_false": "⚠️ Evidence suggests this claim is FALSE. Check the sources above and consult fact-checking websites.",
            "likely_true": "✅ Evidence supports this claim. Verify with additional sources for confirmation.",
            "mixed_evidence": "🔄 Evidence is mixed. The claim may be partially true or context-dependent. Research further.",
            "insufficient_evidence": "❓ Not enough evidence found. Search fact-checking sites like Snopes, PolitiFact, or FactCheck.org."
        }
        return recommendations.get(verdict, "Research this claim further before sharing.")


# Global instance
_retriever = None

def get_retriever() -> EvidenceRetriever:
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever

def retrieve(text: str, max_results: int = 5) -> Dict[str, Any]:
    return get_retriever().retrieve(text, max_results)
