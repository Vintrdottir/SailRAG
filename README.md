# SailRAG — Yacht & Vessel Knowledge Base (OCR + Hybrid Search + Local LLM)

A production-ready RAG demo for yacht/vessel knowledge base:
- PDF text extraction with OCR fallback
- chunking & segmentation
- embeddings via Ollama
- OpenSearch hybrid retrieval (BM25 + vector)
- local LLM inference via Ollama
- Streamlit UI
- fully Dockerized

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

   
## Quickstart
```bash
docker compose up --build

