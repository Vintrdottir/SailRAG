from fastapi import FastAPI
import httpx

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
from fastapi import Body, HTTPException

from sailrag.ingest.models import DocumentPreview, PageText
from sailrag.ingest.pdf_loader import decide_text_vs_ocr, extract_text_for_pages, choose_sample_pages
from sailrag.ingest.ocr import ocr_pdf_pages

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
    Preview document extraction:
    - tries PDF text extraction
    - decides whether to fallback to OCR
    - returns per-page preview text + metadata
    """
    pdf_path = Path(settings.data_dir) / path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")

    # 1) Attempt text-layer extraction
    from sailrag.ingest.pdf_loader import extract_text_for_pages, choose_sample_pages
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    sample_pages = choose_sample_pages(total_pages, max_pages)
    texts, _ = extract_text_for_pages(pdf_path, sample_pages)
    strategy, reason = decide_text_vs_ocr(texts, total_pages)

    pages: list[PageText] = []

    def page_metrics(page_number: int, method: str, text: str) -> PageText:
        char_count = len(text)
        stripped = "".join(text.split())
        ratio = (len(stripped) / char_count) if char_count > 0 else 0.0
        return PageText(
            page_number=page_number,
            method=method,
            text=text,
            char_count=char_count,
            non_whitespace_ratio=ratio,
        )

    if strategy == "text":
        for page_no, t in zip(sample_pages, texts, strict=False):
            pages.append(page_metrics(page_no, "text", t))

        return DocumentPreview(
            path=path,
            pages_total=total_pages,
            pages_previewed=len(pages),
            chosen_strategy="text",
            reason=reason,
            pages=pages,
        )

    # 2) OCR fallback
    ocr_texts, previewed_pages = ocr_pdf_pages(pdf_path, max_pages=max_pages)
    for i, t in enumerate(ocr_texts, start=1):
        pages.append(page_metrics(i, "ocr", t))

    return DocumentPreview(
        path=path,
        pages_total=total_pages,  # from text extraction attempt
        pages_previewed=previewed_pages,
        chosen_strategy="ocr",
        reason=reason,
        pages=pages,
    )
