# SailRAG — Nautical Document Q&A (OCR + Hybrid Search + Local LLM)

SailRAG is a production-style Retrieval-Augmented Generation (RAG) system for querying nautical and maritime documents.
It combines hybrid retrieval (BM25 + vector search), local LLM inference, and page-level citations to provide transparent, grounded answers.

The project is designed as a portfolio-quality, end-to-end system, focusing on correctness, explainability, and realistic tradeoffs rather than API shortcuts.

✨ Key Features

Hybrid Retrieval

BM25 keyword search + dense vector similarity (OpenSearch)

Tunable weighting between lexical and semantic retrieval

Adaptive PDF Ingestion

Page-level decision between text extraction and OCR

Handles mixed documents (text pages + scanned diagrams)

High-Quality Chunking

Windowed chunking with overlap

Table-of-contents detection and filtering

Local Embeddings & LLM

Embeddings: nomic-embed-text via Ollama

Answer generation: local LLM (Ollama, CPU-only)

Transparent Answers

Each answer includes page-level citations

Traceable chunk IDs and document references

Interactive UI

Streamlit app for asking questions

Live control over retrieval parameters

Fully Dockerized

One-command startup

No external paid APIs required

  ## Dataset choice
The system is intentionally limited to English-language public maritime documents (COLREG, sailing manuals, navigation guides) to ensure consistent OCR quality, embedding alignment, and retrieval performance.

1) COLREG — International Regulations for Preventing Collisions at Sea
   https://www.dohle-yachts.com/wp-content/uploads/2022/07/COLREGS-The-Rules-of-the-Road.pdf

2) Basic Sailing Manual (CSUN)
   https://www.csun.edu/sites/default/files/CSUN%20Sailing%20Manual%20updated%202016_0.pdf

3) Sailing Made Simple (full book PDF)
   https://www.sjsu.edu/people/shirley.reekie/courses/sailing/s2/Sailing-Made-Simple-whole-book.pdf

4) Nautical Charts & Navigation Guide
   https://scuba.garykessler.net/library/Charts_Navigation.pdf

5) USCG Navigation Rules (NavRules)
   https://www.navcen.uscg.gov/sites/default/files/pdf/navRules/navrules.pdf



🧭 Architecture Overview
User (Streamlit UI)
        |
        v
FastAPI Backend
  ├── Ingestion & OCR
  ├── Chunking & Metadata
  ├── Embedding Generation (Ollama)
  ├── Hybrid Retrieval (OpenSearch)
  └── Answer Generation (Ollama LLM)
        |
        v
OpenSearch (BM25 + Vector Index)

🧠 Why Hybrid Retrieval?

Pure vector search can miss exact terminology and structured references.
Pure keyword search lacks semantic understanding.

SailRAG uses hybrid retrieval to combine:

BM25 → precise keyword matching (e.g. “hazards”, “bridge clearance”)

kNN vectors → semantic similarity across phrasing

The weights are configurable per query.

🧪 Example Query

Question

How do nautical charts help you avoid hazards in unfamiliar waters?

Answer

Nautical charts provide detailed information about hazards, navigation markers, seabed characteristics, and obstructions such as bridges, enabling safe navigation even in unfamiliar waters.

Citations

Charts_Navigation — page 8 — chunk p8-c3

Charts_Navigation — page 8 — chunk p8-c2

🚀 Running the Project
Requirements

Docker + Docker Compose

~8 GB RAM recommended (CPU-only inference)

Start everything
docker compose up --build


Services:

Streamlit UI → http://localhost:8501

Backend API → http://localhost:8000

OpenSearch → http://localhost:9200

Ollama → http://localhost:11434

🗂️ Project Structure
.
├── backend/
│   └── src/sailrag/
│       ├── ingest/        # PDF loading, OCR, page analysis
│       ├── chunking/      # Chunking + TOC detection
│       ├── embed/         # Embedding generation
│       ├── search/        # Hybrid OpenSearch retrieval
│       ├── llm/           # Ollama LLM client
│       └── main.py        # FastAPI endpoints
├── ui/
│   └── app.py             # Streamlit UI
├── docker-compose.yml
└── README.md

⚙️ Configuration

All configuration is handled via environment variables and typed settings.

Examples:

OPENSEARCH_URL

OLLAMA_URL

OLLAMA_LLM_MODEL

OPENSEARCH_INDEX

🧩 Design Tradeoffs

Local inference

✅ No external API costs

❌ Higher latency than hosted LLMs

Page-level OCR

✅ Accurate handling of mixed PDFs

❌ Slightly slower ingestion

Explicit citations

✅ Trustworthy, debuggable answers

❌ More complex retrieval pipeline

These choices were made intentionally to reflect real production constraints.

🔮 Future Work

Streaming responses from the LLM

Caching embeddings and retrieval results

UI support for document upload

Reranking with cross-encoders

Multi-document summarization

