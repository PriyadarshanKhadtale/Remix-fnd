"""
Evidence Retrieval Feature (RAG)
================================
Retrieves relevant evidence to fact-check claims.
"""

from .retriever import retrieve, EvidenceRetriever

__all__ = ["retrieve", "EvidenceRetriever"]
