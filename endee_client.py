"""
endee_client.py
---------------
HTTP client for the Endee vector database.
Handles: index creation, document upsert, vector search, filtered search.

Endee API reference: https://docs.endee.io
Default server: http://localhost:8080
"""

import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ENDEE_BASE_URL = "http://localhost:8080"
INDEX_NAME = "docmind"
VECTOR_DIM = 384  # matches all-MiniLM-L6-v2 output


class EndeeClient:
    def __init__(self, base_url: str = ENDEE_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------
    # Index Management
    # ------------------------------------------------------------------

    def create_index(self) -> dict:
        """
        Create the vector index in Endee.
        Uses cosine distance which works well for sentence embeddings.
        Idempotent — safe to call if index already exists.
        """
        payload = {
            "name": INDEX_NAME,
            "vectors": {
                "size": VECTOR_DIM,
                "distance": "Cosine"
            }
        }
        response = self.client.put(
            f"{self.base_url}/collections/{INDEX_NAME}",
            json=payload
        )
        if response.status_code in (200, 201):
            logger.info(f"Index '{INDEX_NAME}' ready.")
            return response.json()
        elif response.status_code == 409:
            logger.info(f"Index '{INDEX_NAME}' already exists.")
            return {"status": "exists"}
        else:
            response.raise_for_status()

    def delete_index(self) -> dict:
        """Drop the index (useful for re-seeding from scratch)."""
        response = self.client.delete(
            f"{self.base_url}/collections/{INDEX_NAME}"
        )
        response.raise_for_status()
        return response.json()

    def get_index_info(self) -> dict:
        """Return index metadata (point count, vector config, etc.)."""
        response = self.client.get(
            f"{self.base_url}/collections/{INDEX_NAME}"
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Document Ingestion
    # ------------------------------------------------------------------

    def upsert_documents(self, points: list[dict]) -> dict:
        """
        Upsert a batch of vector points into Endee.

        Each point must have:
          - id: str | int
          - vector: list[float]  (length == VECTOR_DIM)
          - payload: dict        (arbitrary metadata — title, content, category, etc.)

        Endee supports online upsert without reindexing.
        """
        payload = {"points": points}
        response = self.client.put(
            f"{self.base_url}/collections/{INDEX_NAME}/points",
            json=payload
        )
        response.raise_for_status()
        logger.info(f"Upserted {len(points)} points into Endee.")
        return response.json()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Perform vector similarity search.

        Args:
            query_vector:    Embedded query (384-dim float list)
            top_k:           Number of results to return
            category_filter: Optional category string for payload filtering

        Returns:
            List of result dicts with score + payload
        """
        body: dict = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }

        # Endee payload filtering — scopes search to specific categories
        # Docs: https://docs.endee.io (filter section)
        if category_filter:
            body["filter"] = {
                "must": [
                    {
                        "key": "category",
                        "match": {"value": category_filter}
                    }
                ]
            }

        response = self.client.post(
            f"{self.base_url}/collections/{INDEX_NAME}/points/search",
            json=body
        )
        response.raise_for_status()

        raw = response.json()
        results = []
        for hit in raw.get("result", []):
            results.append({
                "id": hit["id"],
                "score": round(hit["score"], 4),
                **hit.get("payload", {})
            })
        return results

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    def health(self) -> bool:
        """Returns True if Endee server is reachable."""
        try:
            response = self.client.get(f"{self.base_url}/healthz", timeout=3.0)
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        self.client.close()
