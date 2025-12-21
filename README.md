# SailRAG — Yacht & Vessel Knowledge Base (OCR + Hybrid Search + Local LLM)

A production-ready RAG demo for yacht/vessel knowledge base:
- PDF text extraction with OCR fallback
- chunking & segmentation
- embeddings via Ollama
- OpenSearch hybrid retrieval (BM25 + vector)
- local LLM inference via Ollama
- Streamlit UI
- fully Dockerized

## Quickstart
```bash
docker compose up --build
