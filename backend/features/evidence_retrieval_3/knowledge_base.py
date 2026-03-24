"""
Knowledge Base Manager
======================
Legacy minimal `KnowledgeBase` class. The **live** EVRS store is
`ExpandedKnowledgeBase` in `retriever.py` (LIAR + hand-crafted facts, FAISS).
Prefer importing from `retriever` for new code.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path


class KnowledgeBase:
    """
    Simple in-memory fact list (legacy). Production path uses ExpandedKnowledgeBase.
    """
    
    def __init__(self):
        self.facts: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
    
    def load(self, path: str) -> None:
        """Load facts from file."""
        # TODO: Implement
        pass
    
    def add_fact(self, fact: Dict[str, Any]) -> None:
        """Add a new fact to the knowledge base."""
        self.facts.append(fact)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant facts."""
        # TODO: Implement semantic search
        return []
    
    def build_index(self) -> None:
        """Build FAISS index for fast retrieval."""
        # TODO: Implement
        pass

