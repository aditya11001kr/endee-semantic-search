# 🔍 DocMind — Semantic Document Search with Endee Vector Database

> A production-style AI application that enables natural language search over a knowledge base using Endee as the vector database backend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Endee](https://img.shields.io/badge/Vector_DB-Endee-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green?style=flat-square&logo=fastapi)

---

## 📌 Project Overview

**DocMind** is a semantic search engine for documents and knowledge bases. Instead of keyword-only matching, it uses sentence embeddings (dense vectors) to understand the *meaning* behind queries — powered by **Endee**, an open-source vector database optimized for AI retrieval workloads.

A user can type something like:
> *"How do I reduce memory usage in Python?"*

…and DocMind will find the most relevant documents — even if they don't contain those exact words.

---

## 🧩 Problem Statement

Traditional search relies on exact keyword matching (BM25, TF-IDF), which fails when:
- The user uses different vocabulary than the document
- Concepts are expressed in different ways across languages
- Context matters more than individual terms

**Solution:** Embed both documents and queries into a shared vector space using a transformer model, then retrieve nearest neighbors using Endee's fast approximate nearest neighbor (ANN) search.

---

## 🏗️ System Design

```
User Query
    │
    ▼
[Embedding Model]  ←── sentence-transformers (all-MiniLM-L6-v2)
    │
    ▼
[Query Vector]
    │
    ▼
[Endee Vector Database]  ←── stores document embeddings + metadata
    │   (ANN search + payload filtering)
    │
    ▼
[Top-K Results with metadata]
    │
    ▼
[FastAPI Backend]  ──→  [Frontend UI]
```

### Components

| Layer | Technology | Role |
|---|---|---|
| Embedding | `sentence-transformers` | Convert text → 384-dim vectors |
| Vector DB | **Endee** | Store & retrieve vectors at scale |
| API | FastAPI | REST endpoints for search & ingest |
| Frontend | HTML/CSS/JS | Clean search UI |
| Data | Custom + Wikipedia snippets | Demo knowledge base |

---

## 🚀 How Endee is Used

Endee serves as the **core retrieval engine** for this project:

1. **Index Creation** — We create an Endee index with 384-dimensional vectors (matching our embedding model's output size)
2. **Document Ingestion** — Each document chunk is embedded and upserted into Endee along with payload metadata (title, source, category)
3. **Semantic Search** — At query time, the query is embedded and sent to Endee's `/search` endpoint; Endee returns the top-K nearest vectors
4. **Filtered Search** — Endee's payload filtering lets users scope search to specific categories (e.g., only search "Python" docs)

Endee replaces what would otherwise require a dedicated FAISS index + a separate metadata store — it handles both vector proximity and structured filtering in one system.

---

## 📁 Project Structure

```
endee-semantic-search/
├── backend/
│   ├── main.py              # FastAPI app — search & ingest endpoints
│   ├── embedder.py          # Sentence transformer wrapper
│   ├── endee_client.py      # Endee HTTP client (index, upsert, search)
│   ├── ingest.py            # Data ingestion pipeline
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file search UI
├── data/
│   └── knowledge_base.json  # Sample documents for demo
├── scripts/
│   ├── install.sh           # Install Endee (wraps upstream install.sh)
│   └── seed_data.sh         # One-command data seeding
├── docs/
│   └── architecture.md      # Extended architecture notes
├── docker-compose.yml       # Run everything with one command
└── README.md
```

---

## ⚙️ Setup and Execution

### Prerequisites
- Python 3.10+
- Docker (optional but recommended)
- 4 GB RAM minimum (for Endee + embedding model)

---

### Option A — Docker (Recommended)

```bash
# 1. Clone this repo
git clone https://github.com/<your-username>/endee-semantic-search.git
cd endee-semantic-search

# 2. Start all services
docker-compose up --build

# 3. Seed the knowledge base
./scripts/seed_data.sh

# 4. Open the UI
open http://localhost:3000
# API docs at http://localhost:8000/docs
```

---

### Option B — Local Setup

**Step 1 — Start Endee**

```bash
# Clone and build Endee (from your forked repo)
git clone https://github.com/<your-username>/endee.git
cd endee
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
# Endee now runs on http://localhost:8080
```

**Step 2 — Set up Python backend**

```bash
cd backend/
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3 — Ingest documents**

```bash
python ingest.py
# Embeds and indexes all documents from ../data/knowledge_base.json
```

**Step 4 — Start the API server**

```bash
uvicorn main:app --reload --port 8000
```

**Step 5 — Open the frontend**

```bash
# Just open frontend/index.html in your browser
# OR serve it:
python -m http.server 3000 --directory ../frontend
open http://localhost:3000
```

---

## 🔌 API Reference

### `POST /search`
Search the knowledge base semantically.

```json
// Request
{
  "query": "how to handle errors in async python",
  "top_k": 5,
  "category_filter": "python"   // optional
}

// Response
{
  "results": [
    {
      "id": "doc_042",
      "title": "Async Exception Handling",
      "content": "When working with asyncio, exceptions...",
      "category": "python",
      "score": 0.91
    }
  ],
  "query_time_ms": 12
}
```

### `POST /ingest`
Add a new document to the index.

```json
{
  "id": "doc_099",
  "title": "My Document",
  "content": "Full text content here...",
  "category": "general"
}
```

### `GET /health`
Returns service health and Endee connection status.

---

## 📊 Demo Knowledge Base

The `data/knowledge_base.json` file includes 50 curated documents across categories:
- **Python** — async, memory, decorators, type hints
- **Machine Learning** — gradient descent, transformers, fine-tuning
- **Databases** — indexing, transactions, NoSQL vs SQL
- **System Design** — caching, load balancing, microservices
- **Algorithms** — sorting, graphs, dynamic programming

---

## 🧠 Why This Architecture Works

| Challenge | Solution |
|---|---|
| Vocabulary mismatch | Semantic embeddings capture meaning, not just words |
| Slow retrieval at scale | Endee's ANN search returns results in milliseconds |
| Category scoping | Endee payload filtering avoids post-processing |
| Adding new docs live | Endee supports online upsert without reindexing |

---

## 🔮 Extensions (Future Work)

- **RAG Pipeline** — pipe Endee results into an LLM for answer generation
- **Sparse + Dense Hybrid** — use Endee's sparse vector support for BM25+semantic hybrid
- **Multi-modal** — embed and search images using CLIP
- **Agent Memory** — use Endee as persistent memory for an LLM agent

---

## 📚 References

- [Endee GitHub](https://github.com/endee-io/endee)
- [Endee Documentation](https://docs.endee.io)
- [sentence-transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 👤 Author

Built as part of the **Endee.io Campus Recruitment Project** — Galgotias University, 2027 Batch.


