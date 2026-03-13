# Architecture Notes — DocMind

## Why Endee?

Endee was chosen as the vector database for three reasons:

1. **Self-hosted control** — Unlike managed services (Pinecone, Weaviate Cloud), Endee runs on your own hardware, giving full data privacy and zero per-query costs.
2. **Payload filtering** — Endee supports filtering search results by arbitrary metadata fields (e.g., `category == "python"`) without a post-processing step. This is critical for category-scoped search.
3. **HTTP API** — Endee's clean REST API makes it easy to integrate from any language/framework without a proprietary SDK dependency.

## Embedding Model Choice

`all-MiniLM-L6-v2` (384 dimensions) was selected because:
- Runs entirely on CPU in <5ms per query
- Strong MTEB benchmark scores for semantic similarity
- Small model size (~80 MB) — fast to download and deploy
- Normalized embeddings work directly with cosine distance

## Data Flow

```
Document text
     │
     ▼
title + content concatenated
     │
     ▼
SentenceTransformer.encode()   →  384-dim float32 vector
     │
     ▼
Endee upsert (vector + payload)
     │
     ▼
Stored in HNSW index inside Endee
```

```
User query string
     │
     ▼
SentenceTransformer.encode()   →  384-dim query vector
     │
     ▼
Endee /search  (cosine ANN + optional payload filter)
     │
     ▼
Top-K result payloads  →  FastAPI response  →  Frontend
```

## Endee Index Configuration

- **Distance metric:** Cosine — correct for normalized sentence embeddings
- **Vector size:** 384 — must match the embedding model's output dimension
- **Index type:** HNSW (default in Endee) — best recall/performance tradeoff

## Scaling Considerations

| Scale | Approach |
|---|---|
| < 100K docs | Single Endee node, default settings |
| 100K – 1M docs | Increase HNSW `m` and `ef_construct` parameters |
| > 1M docs | Endee distributed mode or shard across collections |

## Extending to Full RAG

To turn DocMind into a full RAG pipeline:
1. Take the top-K retrieved chunks from Endee
2. Concatenate them as context into an LLM prompt
3. Ask the LLM to answer the query using only the provided context
4. Return both the answer and source citations

This pattern reduces hallucination by grounding the LLM in real documents.
