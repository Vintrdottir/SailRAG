from __future__ import annotations

from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def ocr_pdf_pages(
    pdf_path: Path,
    max_pages: int | None = None,
    dpi: int = 200,
) -> tuple[list[str], int]:
    """
    Render PDF pages to images (via poppler) and run Tesseract OCR.
    Used mainly for batch preview or legacy paths.

    Returns:
        (texts, pages_rendered)
    """
    first_page = 1
    last_page = max_pages if max_pages is not None else None

    images: list[Image.Image] = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt="png",
    )

    texts: list[str] = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang="eng")
        texts.append(txt)

    return texts, len(images)


def ocr_pdf_page(
    pdf_path: Path,
    page_number_1based: int,
    dpi: int = 200,
) -> str:
    """
    Render exactly ONE PDF page (1-based index) to image and OCR it.
    Used by page-level adaptive ingestion.
    """
    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number_1based,
        last_page=page_number_1based,
        fmt="png",
    )

    if not images:
        return ""

    return pytesseract.image_to_string(images[0], lang="eng")