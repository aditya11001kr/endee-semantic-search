"""
embedder.py
-----------
Thin wrapper around sentence-transformers.
Uses all-MiniLM-L6-v2 — a lightweight 384-dim model that's fast enough
for real-time queries and accurate enough for most semantic search tasks.
"""

from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded.")

    def embed(self, text: str) -> list[float]:
        """Embed a single string → 384-dim float list."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in one forward pass (faster than loop)."""
        vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in vectors]
