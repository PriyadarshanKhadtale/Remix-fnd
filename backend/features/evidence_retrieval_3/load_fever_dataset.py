"""
Load FEVER Dataset for Evidence Retrieval
==========================================
FEVER = Fact Extraction and VERification (20K+ claims)
Wikipedia-based fact verification dataset.

**Optional developer path.** The default `EvidenceRetriever` / `ExpandedKnowledgeBase`
loads the **LIAR**-based KB from `retriever.py`. FEVER is not wired into that default
pipeline; use this module only if you build a custom KB loader. See repository `SCOPE.md`.
"""

import json
from typing import List, Dict
from pathlib import Path


# Label mapping
LABEL_TO_STANCE = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes", 
    "NOT ENOUGH INFO": "neutral"
}


def load_fever_dataset(data_dir: str = None, max_samples: int = None) -> List[Dict]:
    """
    Load FEVER dataset and convert to knowledge base format.
    
    Args:
        data_dir: Path to directory containing fever_sample.jsonl
        max_samples: Maximum samples to load
    
    Returns:
        List of fact entries in KB format
    """
    if data_dir is None:
        # Try multiple paths
        possible_paths = [
            Path("/app/data/fact_checking/fever"),  # Docker
            Path(__file__).parent.parent.parent / "data_fact_checking" / "fever",  # Backend
            Path(__file__).parent.parent.parent.parent / "data" / "fact_checking" / "fever",  # Project
        ]
        data_dir = None
        for p in possible_paths:
            if p.exists():
                data_dir = p
                break
        
        if data_dir is None:
            print("  ⚠️ FEVER dataset directory not found")
            return []
    else:
        data_dir = Path(data_dir)
    
    # Find jsonl file
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  ⚠️ No JSONL files found in {data_dir}")
        return []
    
    facts = []
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and len(facts) >= max_samples:
                    break
                
                try:
                    entry = json.loads(line)
                    
                    claim = entry.get('claim', '')
                    label = entry.get('label', 'NOT ENOUGH INFO')
                    verifiable = entry.get('verifiable', 'NOT VERIFIABLE')
                    
                    if not claim:
                        continue
                    
                    # Skip unverifiable claims for cleaner KB
                    if verifiable == 'NOT VERIFIABLE':
                        continue
                    
                    stance = LABEL_TO_STANCE.get(label, 'neutral')
                    
                    # Extract keywords
                    words = claim.lower().split()
                    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                                'have', 'has', 'had', 'do', 'does', 'did', 'to', 'of',
                                'in', 'for', 'on', 'with', 'at', 'by', 'from', 'that',
                                'this', 'it', 'and', 'or', 'but', 'not', 'no'}
                    keywords = [w for w in words if len(w) > 3 and w not in stopwords][:10]
                    
                    # Build content
                    content = f"Claim: \"{claim}\" - Verdict: {label}"
                    
                    facts.append({
                        "id": f"fever_{entry.get('id', i)}",
                        "category": "wikipedia",
                        "title": claim[:100] + "..." if len(claim) > 100 else claim,
                        "content": content,
                        "source": "FEVER Dataset (Wikipedia)",
                        "stance": stance,
                        "credibility": 1.0 if stance == "supports" else (0.0 if stance == "refutes" else 0.5),
                        "keywords": keywords,
                        "verifiable": verifiable
                    })
                    
                except json.JSONDecodeError:
                    continue
    
    print(f"  ✓ Loaded {len(facts)} claims from FEVER dataset")
    return facts


if __name__ == "__main__":
    facts = load_fever_dataset(max_samples=100)
    print(f"\nLoaded {len(facts)} facts")
    
    if facts:
        print("\nSample fact:")
        import json as j
        print(j.dumps(facts[0], indent=2))
        
        # Show stance distribution
        stances = {}
        for f in facts:
            s = f.get("stance", "unknown")
            stances[s] = stances.get(s, 0) + 1
        print("\nStance distribution:")
        for s, c in stances.items():
            print(f"  {s}: {c}")

