from __future__ import annotations

from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def ocr_pdf_pages(pdf_path: Path, max_pages: int | None = None, dpi: int = 200) -> tuple[list[str], int]:
    """
    Render PDF pages to images (via poppler) and run Tesseract OCR.
    Returns (texts, total_pages).
    """
    # pdf2image doesn't easily give total pages without rendering; but we can render with a limit.
    # We'll treat rendered pages as "previewed pages" and separately compute total pages using pypdf in caller if needed.
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
        # Good default config for English documents; we can tune later
        txt = pytesseract.image_to_string(img, lang="eng")
        texts.append(txt)

    return texts, len(images)
