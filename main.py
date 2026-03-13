"""
main.py
-------
FastAPI application exposing:
  POST /search  — semantic search over the knowledge base
  POST /ingest  — add a single document to the index
  GET  /health  — service health + Endee connection check
  GET  /stats   — index statistics
"""

import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from embedder import Embedder
from endee_client import EndeeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocMind — Semantic Search API",
    description="Semantic document search powered by Endee vector database",
    version="1.0.0"
)

# Allow frontend (served on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources at startup
embedder = Embedder()
endee = EndeeClient()


# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "how to handle errors in async python",
                "top_k": 5,
                "category_filter": "python"
            }
        }


class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    category: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_time_ms: int
    total: int


class IngestRequest(BaseModel):
    id: str
    title: str
    content: str
    category: str = "general"


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health_check():
    endee_alive = endee.health()
    return {
        "status": "ok" if endee_alive else "degraded",
        "endee_connected": endee_alive,
        "embedding_model": "all-MiniLM-L6-v2"
    }


@app.get("/stats")
def index_stats():
    try:
        info = endee.get_index_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Endee unavailable: {e}")


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = time.monotonic()

    # 1. Embed the query
    query_vector = embedder.embed(req.query)

    # 2. Search Endee
    try:
        hits = endee.search(
            query_vector=query_vector,
            top_k=req.top_k,
            category_filter=req.category_filter
        )
    except Exception as e:
        logger.error(f"Endee search failed: {e}")
        raise HTTPException(status_code=503, detail="Vector database error.")

    elapsed_ms = int((time.monotonic() - start) * 1000)

    results = [
        SearchResult(
            id=str(h["id"]),
            title=h.get("title", "Untitled"),
            content=h.get("content", ""),
            category=h.get("category", "general"),
            score=h["score"]
        )
        for h in hits
    ]

    return SearchResponse(
        results=results,
        query_time_ms=elapsed_ms,
        total=len(results)
    )


@app.post("/ingest")
def ingest(req: IngestRequest):
    # Embed document content
    vector = embedder.embed(req.title + " " + req.content)

    point = {
        "id": req.id,
        "vector": vector,
        "payload": {
            "title": req.title,
            "content": req.content,
            "category": req.category
        }
    }

    try:
        result = endee.upsert_documents([point])
    except Exception as e:
        logger.error(f"Endee upsert failed: {e}")
        raise HTTPException(status_code=503, detail="Failed to index document.")

    return {"status": "indexed", "id": req.id}
