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
        page_chunks = chunk_text_windowed(page.text, max_chars=max_chars, overlap=overlap, min_chars=min_chars)
        for idx, ch in enumerate(page_chunks, start=1):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    page_number=page.page_number,
                    chunk_id=f"{doc_id}-p{page.page_number}-c{idx}",
                    text=ch,
                    char_count=len(ch),
                )
            )

    return {
        "doc_id": doc_id,
        "pages_previewed": preview.pages_previewed,
        "chunks_total": len(chunks),
        "chunks": [c.model_dump() for c in chunks[:12]],  # show first 12 for preview
    }


