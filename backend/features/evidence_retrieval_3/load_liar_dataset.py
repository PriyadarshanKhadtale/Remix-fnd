"""
Load LIAR Dataset into Knowledge Base
=====================================
Processes the LIAR fact-checking dataset (12.8K political claims)
into our knowledge base format for evidence retrieval.
"""

import os
import csv
from typing import List, Dict
from pathlib import Path


# Label mapping
LABEL_TO_STANCE = {
    "true": "verified_true",
    "mostly-true": "mostly_true", 
    "half-true": "mixed",
    "barely-true": "mostly_false",
    "false": "verified_false",
    "pants-fire": "pants_on_fire"  # Completely false
}

LABEL_TO_CREDIBILITY = {
    "true": 1.0,
    "mostly-true": 0.8,
    "half-true": 0.5,
    "barely-true": 0.3,
    "false": 0.1,
    "pants-fire": 0.0
}


def load_liar_dataset(data_dir: str = None) -> List[Dict]:
    """
    Load LIAR dataset and convert to knowledge base format.
    
    Args:
        data_dir: Path to directory containing train.tsv, valid.tsv, test.tsv
    
    Returns:
        List of fact entries in KB format
    """
    if data_dir is None:
        # Try multiple paths (Docker vs local)
        possible_paths = [
            Path("/app/data/fact_checking"),  # Docker path
            Path(__file__).parent.parent.parent / "data_fact_checking",  # Backend local
            Path(__file__).parent.parent.parent.parent / "data" / "fact_checking",  # Project root
        ]
        data_dir = None
        for p in possible_paths:
            if p.exists() and (p / "train.tsv").exists():
                data_dir = p
                break
        
        if data_dir is None:
            print("  ⚠️ LIAR dataset not found in any known location")
            return []
    else:
        data_dir = Path(data_dir)
    
    facts = []
    
    # Load all splits
    for split in ["train.tsv", "valid.tsv", "test.tsv"]:
        filepath = data_dir / split
        if not filepath.exists():
            print(f"  Warning: {filepath} not found")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) < 14:
                    continue
                
                try:
                    # Parse LIAR format
                    # Columns: id, label, statement, subject, speaker, job, state, party, 
                    #          barely_true_counts, false_counts, half_true_counts, mostly_true_counts,
                    #          pants_on_fire_counts, context
                    
                    claim_id = row[0]
                    label = row[1].strip().lower()
                    statement = row[2].strip()
                    subject = row[3].strip() if row[3] else "general"
                    speaker = row[4].strip() if row[4] else "unknown"
                    job = row[5].strip() if row[5] else ""
                    state = row[6].strip() if row[6] else ""
                    party = row[7].strip() if row[7] else ""
                    context = row[13].strip() if len(row) > 13 and row[13] else ""
                    
                    if not statement or label not in LABEL_TO_STANCE:
                        continue
                    
                    # Build source attribution
                    source_parts = [speaker]
                    if job:
                        source_parts.append(f"({job})")
                    if party:
                        source_parts.append(f"- {party}")
                    source = " ".join(source_parts) if source_parts else "Unknown"
                    
                    # Build content with context
                    content = f"Claim: \"{statement}\""
                    if context:
                        content += f" (Said in: {context})"
                    content += f" Verdict: {label.upper().replace('-', ' ')}"
                    
                    # Extract keywords from statement
                    words = statement.lower().split()
                    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                                'that', 'this', 'it', 'its', 'and', 'or', 'but', 'not', 'no'}
                    keywords = [w for w in words if len(w) > 3 and w not in stopwords][:10]
                    
                    # Add subject as keyword
                    if subject and subject != "general":
                        keywords.extend(subject.lower().split(','))
                    
                    facts.append({
                        "id": claim_id,
                        "category": subject.split(',')[0] if subject else "politics",
                        "title": statement[:100] + "..." if len(statement) > 100 else statement,
                        "content": content,
                        "source": f"PolitiFact - {source}",
                        "stance": LABEL_TO_STANCE.get(label, "unknown"),
                        "credibility": LABEL_TO_CREDIBILITY.get(label, 0.5),
                        "keywords": list(set(keywords)),
                        "speaker": speaker,
                        "party": party,
                        "context": context
                    })
                    
                except Exception as e:
                    continue
    
    print(f"  Loaded {len(facts)} claims from LIAR dataset")
    return facts


def get_liar_facts() -> List[Dict]:
    """Get LIAR dataset facts (cached)."""
    return load_liar_dataset()


if __name__ == "__main__":
    # Test loading
    facts = load_liar_dataset()
    print(f"\nTotal facts loaded: {len(facts)}")
    
    if facts:
        print("\nSample fact:")
        import json
        print(json.dumps(facts[0], indent=2))
        
        print("\nCategory distribution:")
        categories = {}
        for f in facts:
            cat = f.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cat}: {count}")

