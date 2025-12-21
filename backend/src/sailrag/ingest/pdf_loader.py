from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class TextQuality:
    char_count: int
    non_whitespace_ratio: float


def _quality(text: str) -> TextQuality:
    stripped = "".join(text.split())
    char_count = len(text)
    ratio = (len(stripped) / char_count) if char_count > 0 else 0.0
    return TextQuality(char_count=char_count, non_whitespace_ratio=ratio)


def extract_text_for_pages(pdf_path: Path, page_numbers_1based: list[int]) -> tuple[list[str], int]:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    texts: list[str] = []
    for p in page_numbers_1based:
        if p < 1 or p > total_pages:
            texts.append("")
            continue
        page = reader.pages[p - 1]
        texts.append(page.extract_text() or "")
    return texts, total_pages


def choose_sample_pages(total_pages: int, max_pages: int) -> list[int]:
    """
    Pick pages across the document to avoid TOC/cover bias.
    Returns 1-based page numbers.
    """
    if total_pages <= 0:
        return []

    # Always include page 1
    pages = [1]

    if total_pages >= 2 and max_pages >= 2:
        mid = (total_pages // 2) or 1
        if mid not in pages:
            pages.append(mid)

    if total_pages >= 3 and max_pages >= 3:
        if total_pages not in pages:
            pages.append(total_pages)

    # If max_pages > 3, fill with evenly spaced pages
    while len(pages) < max_pages and total_pages > 1:
        # pick next evenly spaced page
        step = max(1, total_pages // (max_pages + 1))
        candidate = 1 + step * (len(pages))
        candidate = min(total_pages, max(1, candidate))
        if candidate not in pages:
            pages.append(candidate)
        else:
            # fallback: just increment
            nxt = min(total_pages, pages[-1] + 1)
            if nxt not in pages:
                pages.append(nxt)
            else:
                break

    return sorted(pages)[:max_pages]


def decide_text_vs_ocr(page_texts: list[str], total_pages: int) -> tuple[str, str]:
    """
    Heuristic strategy:
    - If enough pages have meaningful text, choose "text"
    - Else choose "ocr"
    """
    if total_pages == 0:
        return "text", "Empty PDF (0 pages)."

    # Aggregate quality across previewed pages (not necessarily all pages)
    qualities = [_quality(t) for t in page_texts]
    total_chars = sum(q.char_count for q in qualities)
    avg_ratio = (sum(q.non_whitespace_ratio for q in qualities) / len(qualities)) if qualities else 0.0

    # Also: count how many pages look "non-empty"
    non_empty_pages = sum(1 for q in qualities if q.char_count >= 200 and q.non_whitespace_ratio >= 0.10)

    # Thresholds tuned for manuals/regulations; we'll adjust after seeing your PDFs in preview
    if total_chars >= 1200 and non_empty_pages >= max(1, len(page_texts) // 3):
        return "text", f"Text layer seems present (chars={total_chars}, non_empty_pages={non_empty_pages}, avg_ratio={avg_ratio:.2f})."

    return "ocr", f"Text layer seems weak (chars={total_chars}, non_empty_pages={non_empty_pages}, avg_ratio={avg_ratio:.2f})."
