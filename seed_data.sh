#!/usr/bin/env bash
# seed_data.sh
# Runs the ingestion pipeline to embed + index all documents into Endee.
# Call this once after starting the services.

set -e

echo "──────────────────────────────────────"
echo "  DocMind — Data Seeding Script"
echo "──────────────────────────────────────"

# If running against Docker, exec into the api container
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "docmind-api"; then
  echo "Detected Docker environment — running inside docmind-api container..."
  docker exec docmind-api python ingest.py "$@"
else
  echo "Running locally (ensure Endee is running on :8080)..."
  cd "$(dirname "$0")/../backend"
  python ingest.py "$@"
fi

echo ""
echo "✅ Seeding complete. Open http://localhost:3000 to search."
