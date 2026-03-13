"""
ingest.py
---------
One-time script to:
  1. Create the Endee index (idempotent)
  2. Load documents from ../data/knowledge_base.json
  3. Batch-embed all documents
  4. Upsert into Endee with full metadata payload

Run once before starting the API server:
  python ingest.py

To re-seed from scratch (drops existing data):
  python ingest.py --reset
"""

import json
import sys
import logging
from pathlib import Path

from embedder import Embedder
from endee_client import EndeeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "knowledge_base.json"
BATCH_SIZE = 32  # number of documents to upsert per API call


def main():
    reset = "--reset" in sys.argv

    endee = EndeeClient()
    embedder = Embedder()

    # Check Endee is running
    if not endee.health():
        logger.error(
            "Endee server is not reachable at http://localhost:8080.\n"
            "Start Endee first: ./run.sh (see README for setup steps)."
        )
        sys.exit(1)

    # Optionally drop + recreate index
    if reset:
        logger.info("--reset flag detected. Dropping existing index...")
        try:
            endee.delete_index()
            logger.info("Index dropped.")
        except Exception:
            logger.info("No existing index to drop.")

    # Create index (idempotent)
    endee.create_index()

    # Load documents
    logger.info(f"Loading documents from {DATA_PATH}")
    with open(DATA_PATH) as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents.")

    # Batch embed + upsert
    texts = [doc["title"] + " " + doc["content"] for doc in documents]

    logger.info("Embedding all documents (this may take a minute)...")
    all_vectors = embedder.embed_batch(texts)
    logger.info("Embedding complete.")

    # Build Endee point objects
    points = []
    for doc, vector in zip(documents, all_vectors):
        points.append({
            "id": doc["id"],
            "vector": vector,
            "payload": {
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"]
            }
        })

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        endee.upsert_documents(batch)
        total_upserted += len(batch)
        logger.info(f"  Upserted {total_upserted}/{len(points)} documents...")

    logger.info(f"\n✅ Ingestion complete — {total_upserted} documents indexed in Endee.")
    logger.info("You can now start the API: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()
