from fastapi import FastAPI
import httpx
from sailrag.chunking.chunker import chunk_text_windowed
from sailrag.chunking.models import Chunk
from sailrag.settings import settings

app = FastAPI(title="SailRAG API", version="0.1.0")


@app.get("/healthz")
async def healthz():
    """
    Health endpoint checks basic reachability of dependencies.
    This is not a full readiness check, but good enough for compose gating.
    """
    async with httpx.AsyncClient(timeout=3.0) as client:
        os_ok = False
        ollama_ok = False

        try:
            r = await client.get(f"{settings.opensearch_url}")
            os_ok = r.status_code == 200
        except Exception:
            os_ok = False

        try:
            r = await client.get(f"{settings.ollama_url}/api/tags")
            ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

    return {
        "status": "ok" if (os_ok and ollama_ok) else "degraded",
        "opensearch_ok": os_ok,
        "ollama_ok": ollama_ok,
        "env": settings.app_env,
    }

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from pypdf import PdfReader

from sailrag.ingest.models import DocumentPreview, DocumentPreviewSummary, PageText
from sailrag.ingest.pdf_loader import extract_text_for_page, is_text_good_enough, _quality
from sailrag.ingest.ocr import ocr_pdf_page

@app.get("/ingest/list")
async def ingest_list():
    base = Path(settings.data_dir) / "raw_pdfs"
    if not base.exists():
        return {"base": str(base), "pdfs": []}

    pdfs = sorted([p.name for p in base.glob("*.pdf")])
    return {"base": str(base), "pdfs": pdfs}


@app.post("/ingest/preview", response_model=DocumentPreview)
async def ingest_preview(
    path: str = Body(..., embed=True),
    max_pages: int = Body(3, embed=True, ge=1, le=20),
):
    """
    Preview document extraction with PAGE-LEVEL adaptive fallback:
    - for each page in the preview range:
      - try PDF text extraction
      - if text quality is weak -> OCR that page
    """
    pdf_path = Path(settings.data_dir) / path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    preview_pages = min(max_pages, total_pages)

    pages: list[PageText] = []
    text_pages = 0
    ocr_pages = 0

    for page_no in range(1, preview_pages + 1):
        extracted = extract_text_for_page(pdf_path, page_no)

        if is_text_good_enough(extracted):
            method = "text"
            text = extracted
            text_pages += 1
        else:
            method = "ocr"
            text = ocr_pdf_page(pdf_path, page_no)
            ocr_pages += 1

        q = _quality(text)
        pages.append(
            PageText(
                page_number=page_no,
                method=method,
                text=text,
                char_count=q.char_count,
                non_whitespace_ratio=q.non_whitespace_ratio,
            )
        )

    return DocumentPreview(
        path=path,
        pages_total=total_pages,
        pages_previewed=len(pages),
        summary=DocumentPreviewSummary(text_pages=text_pages, ocr_pages=ocr_pages),
        pages=pages,
    )
    
from sailrag.chunking.chunker import chunk_text_windowed, looks_like_table_of_contents

@app.post("/chunk/preview")
async def chunk_preview(
    path: str = Body(..., embed=True),
    max_pages: int = Body(3, embed=True, ge=1, le=30),
    max_chars: int = Body(900, embed=True, ge=200, le=3000),
    overlap: int = Body(150, embed=True, ge=0, le=500),
    min_chars: int = Body(120, embed=True, ge=20, le=500),

):
    """
    Run ingestion preview + chunking preview (no indexing yet).
    Returns chunk examples for debugging.
    """
    # reuse the existing ingestion preview logic by calling the function directly
    preview = await ingest_preview(path=path, max_pages=max_pages)

    # derive doc_id from filename
    doc_id = Path(path).name.replace(".pdf", "")

    chunks: list[Chunk] = []
    for page in preview.pages:
        is_toc = looks_like_table_of_contents(page.text)
        tags = ["toc"] if is_toc else []

        page_chunks = chunk_text_windowed(
            page.text,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
        )

        for idx, ch in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{doc_id}-p{page.page_number}-c{idx}",
                    text=ch,
                    char_count=len(ch),
                    tags=tags,
                )
            )

    non_toc = [c for c in chunks if "toc" not in c.tags]

    return {
        "doc_id": doc_id,
        "pages_previewed": preview.pages_previewed,
        "chunks_total": len(chunks),
        "chunks_non_toc_total": len(non_toc),
        "chunks": [c.model_dump() for c in non_toc[:3]],  # show first 3 non-TOC chunks
    }

from sailrag.embeddings.ollama import embed_text_ollama

@app.post("/embed/preview")
async def embed_preview(
    text: str = Body(..., embed=True),
):
    vec = await embed_text_ollama(
        ollama_url=settings.ollama_url,
        model=settings.ollama_embed_model,
        text=text,
    )
    return {"dim": len(vec), "vector_head": vec[:8]}


import time

@app.post("/embed/chunks_preview")
async def embed_chunks_preview(
    path: str = Body(..., embed=True),
    max_pages: int = Body(3, embed=True, ge=1, le=30),
    max_chars: int = Body(900, embed=True, ge=200, le=3000),
    overlap: int = Body(150, embed=True, ge=0, le=500),
    min_chars: int = Body(120, embed=True, ge=20, le=500),
    max_chunks: int = Body(12, embed=True, ge=1, le=100),
):
    """
    Debug endpoint: ingest -> chunk -> (filter TOC) -> embed first N chunks.
    No indexing yet.
    """
    t0 = time.time()

    # 1) Ingest preview (page-level adaptive)
    preview = await ingest_preview(path=path, max_pages=max_pages)

    doc_id = Path(path).name.replace(".pdf", "")

    # 2) Chunk preview and TOC filtering (reuse your logic)
    chunks: list[Chunk] = []
    for page in preview.pages:
        is_toc = looks_like_table_of_contents(page.text)
        tags = ["toc"] if is_toc else []

        page_chunks = chunk_text_windowed(
            page.text,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
        )

        for idx, ch in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{doc_id}-p{page.page_number}-c{idx}",
                    text=ch,
                    char_count=len(ch),
                    tags=tags,
                )
            )

    non_toc = [c for c in chunks if "toc" not in c.tags]
    to_embed = non_toc[:max_chunks]

    # 3) Embeddings (sequential for now; later we can batch/parallelize)
    vectors_head: list[list[float]] = []
    for c in to_embed:
        vec = await embed_text_ollama(
            ollama_url=settings.ollama_url,
            model=settings.ollama_embed_model,
            text=c.text,
        )
        vectors_head.append(vec[:8])

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "doc_id": doc_id,
        "pages_previewed": preview.pages_previewed,
        "chunks_total": len(chunks),
        "chunks_non_toc_total": len(non_toc),
        "embedded_chunks": len(to_embed),
        "embedding_dim": 768,  # we can also compute from first vec if you want
        "elapsed_ms": elapsed_ms,
        "examples": [
            {
                "chunk_id": c.chunk_id,
                "page_number": c.page_number,
                "char_count": c.char_count,
                "vector_head": vectors_head[i],
            }
            for i, c in enumerate(to_embed)
        ],
    }


from sailrag.opensearch.index import ensure_index

@app.post("/index/create")
async def index_create():
    return await ensure_index(
        opensearch_url=settings.opensearch_url,
        index_name=settings.opensearch_index,
        embedding_dim=768,
    )


from sailrag.opensearch.client import bulk_index

@app.post("/index/ingest")
async def index_ingest(
    path: str = Body(..., embed=True),
    max_pages: int = Body(30, embed=True, ge=1, le=300),
    max_chars: int = Body(900, embed=True, ge=200, le=3000),
    overlap: int = Body(150, embed=True, ge=0, le=500),
    min_chars: int = Body(120, embed=True, ge=20, le=500),
):
    # Ensure index exists
    await ensure_index(settings.opensearch_url, settings.opensearch_index, embedding_dim=768)

    preview = await ingest_preview(path=path, max_pages=max_pages)
    doc_id = Path(path).name.replace(".pdf", "")

    # Chunk + filter TOC
    chunks: list[Chunk] = []
    for page in preview.pages:
        is_toc = looks_like_table_of_contents(page.text)
        tags = ["toc"] if is_toc else []

        page_chunks = chunk_text_windowed(
            page.text,
            max_chars=max_chars,
            overlap=overlap,
            min_chars=min_chars,
        )

        for idx, ch in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{doc_id}-p{page.page_number}-c{idx}",
                    text=ch,
                    char_count=len(ch),
                    tags=tags,
                )
            )

    non_toc = [c for c in chunks if "toc" not in c.tags]

    # Embed + prepare bulk docs
    bulk_docs: list[dict] = []
    for c in non_toc:
        vec = await embed_text_ollama(
            ollama_url=settings.ollama_url,
            model=settings.ollama_embed_model,
            text=c.text,
        )
        bulk_docs.append(
            {
                "id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "page_number": c.page_number,
                "tags": c.tags,
                "text": c.text,
                "embedding": vec,
            }
        )

    bulk_res = await bulk_index(settings.opensearch_url, settings.opensearch_index, bulk_docs)

    return {
        "doc_id": doc_id,
        "pages_indexed": preview.pages_previewed,
        "chunks_total": len(chunks),
        "chunks_indexed": len(non_toc),
        "bulk": bulk_res,
    }



