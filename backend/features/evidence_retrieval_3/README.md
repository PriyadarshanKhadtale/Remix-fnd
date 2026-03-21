# Evidence Retrieval Feature (RAG)

## Overview

This feature uses Retrieval-Augmented Generation (RAG) to fact-check claims by retrieving relevant evidence from a knowledge base.

## Status: 🚧 Not Yet Implemented

## How It Works

```
Claim Input → Embedding → Vector Search → Re-ranking → Evidence → Verdict
                              ↓
                      Knowledge Base (FAISS)
```

## Planned Features

| Feature | Description |
|---------|-------------|
| Semantic Search | Find relevant facts using embeddings |
| Source Verification | Check credibility of sources |
| Contradiction Detection | Identify conflicting information |
| Confidence Scoring | Rate how well evidence supports/refutes claim |

## Knowledge Base Sources (Planned)

- Wikipedia dumps
- Fact-checking websites (Snopes, PolitiFact)
- News archives
- Official government data

## Requirements to Implement

1. Build knowledge base from trusted sources
2. Create embeddings using sentence-transformers
3. Build FAISS index for fast retrieval
4. Implement re-ranking model
5. Add verdict generation logic

