"""
Load HC3 Dataset for AI Detection Training
===========================================
HC3 = Human-ChatGPT Comparison Dataset (24K+ QA pairs)
Contains both human and ChatGPT answers for comparison.
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_hc3_dataset(data_dir: str = None, max_samples: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load HC3 dataset and return human vs AI text samples.
    
    Args:
        data_dir: Path to directory containing hc3_sample.json
        max_samples: Maximum samples to load (None = all)
    
    Returns:
        Tuple of (human_samples, ai_samples)
    """
    if data_dir is None:
        # Try multiple paths (Docker vs local)
        possible_paths = [
            Path("/app/data/ai_detection"),  # Docker path
            Path(__file__).parent.parent.parent / "data_ai_detection",  # Backend local
            Path(__file__).parent.parent.parent.parent / "data" / "ai_detection",  # Project root
        ]
        data_dir = None
        for p in possible_paths:
            if p.exists() and (p / "hc3_sample.json").exists():
                data_dir = p
                break
        
        if data_dir is None:
            print("  ⚠️ HC3 dataset not found in any known location")
            return [], []
    else:
        data_dir = Path(data_dir)
    
    filepath = data_dir / "hc3_sample.json"
    if not filepath.exists():
        print(f"  ⚠️ HC3 dataset not found at {filepath}")
        return [], []
    
    human_samples = []
    ai_samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                entry = json.loads(line)
                question = entry.get('question', '')
                source = entry.get('source', 'unknown')
                
                # Extract human answers
                for answer in entry.get('human_answers', []):
                    if answer and len(answer) > 50:  # Minimum length
                        human_samples.append({
                            'text': answer,
                            'label': 'human',
                            'source': source,
                            'question': question[:100]
                        })
                
                # Extract ChatGPT answers
                for answer in entry.get('chatgpt_answers', []):
                    if answer and len(answer) > 50:
                        ai_samples.append({
                            'text': answer,
                            'label': 'ai',
                            'source': source,
                            'question': question[:100]
                        })
                        
            except json.JSONDecodeError:
                continue
    
    print(f"  ✓ Loaded HC3: {len(human_samples)} human + {len(ai_samples)} AI samples")
    return human_samples, ai_samples


def get_training_data(max_samples: int = 10000) -> List[Dict]:
    """Get balanced training data for AI detection."""
    human, ai = load_hc3_dataset(max_samples=max_samples // 2)
    
    # Balance the dataset
    min_size = min(len(human), len(ai))
    
    data = []
    data.extend(human[:min_size])
    data.extend(ai[:min_size])
    
    return data


def compute_text_features(text: str) -> Dict:
    """
    Compute features from text that distinguish human vs AI.
    Based on research findings about AI text characteristics.
    """
    import re
    import string
    from collections import Counter
    
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not words or not sentences:
        return {}
    
    # 1. Sentence length statistics
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    sent_len_var = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
    
    # 2. Word length statistics
    word_lengths = [len(w) for w in words]
    avg_word_len = sum(word_lengths) / len(word_lengths)
    
    # 3. Vocabulary richness
    unique_words = set(words)
    ttr = len(unique_words) / len(words)  # Type-Token Ratio
    
    # 4. Punctuation usage
    punct_count = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_count / len(text)
    
    # 5. Common AI phrases
    ai_phrases = [
        'it is important to note',
        'in conclusion',
        'furthermore',
        'additionally',
        'it should be noted',
        'there are several',
        'one of the',
        'in order to',
        'as a result',
        'on the other hand'
    ]
    ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text.lower())
    
    # 6. Personal pronouns (humans use more)
    personal_pronouns = ['i', 'me', 'my', 'we', 'us', 'our']
    pronoun_count = sum(1 for w in words if w in personal_pronouns)
    pronoun_ratio = pronoun_count / len(words)
    
    # 7. Contractions (humans use more)
    contractions = ["'t", "'s", "'re", "'ve", "'ll", "'d", "n't"]
    contraction_count = sum(1 for c in contractions if c in text.lower())
    
    return {
        'avg_sentence_length': avg_sent_len,
        'sentence_length_variance': sent_len_var,
        'avg_word_length': avg_word_len,
        'type_token_ratio': ttr,
        'punctuation_ratio': punct_ratio,
        'ai_phrase_count': ai_phrase_count,
        'personal_pronoun_ratio': pronoun_ratio,
        'contraction_count': contraction_count,
        'num_sentences': len(sentences),
        'num_words': len(words)
    }


if __name__ == "__main__":
    # Test loading
    human, ai = load_hc3_dataset(max_samples=100)
    print(f"\nLoaded: {len(human)} human, {len(ai)} AI samples")
    
    if human and ai:
        print("\n--- Human sample features ---")
        features_h = compute_text_features(human[0]['text'])
        for k, v in features_h.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        
        print("\n--- AI sample features ---")
        features_ai = compute_text_features(ai[0]['text'])
        for k, v in features_ai.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

