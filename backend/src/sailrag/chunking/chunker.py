from __future__ import annotations

import re

def looks_like_table_of_contents(text: str) -> bool:
    t = (text or "").lower()
    if "table of contents" in t or t.strip().startswith("contents"):
        return True

    # Heuristic: lots of dot leaders + many short lines ending with numbers
    dot_leaders = t.count("...")  # crude but works well
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    ends_with_number = sum(1 for ln in lines if ln and ln[-1].isdigit())

    if dot_leaders >= 5 and ends_with_number >= 5:
        return True

    # Another heuristic: many lines with tabs and trailing numbers
    if sum(1 for ln in lines if "\t" in ln and any(ch.isdigit() for ch in ln[-6:])) >= 8:
        return True

    return False


def normalize_pdf_text(text: str) -> str:
    """
    Normalize common PDF extraction artifacts:
    - remove excessive whitespace
    - collapse lines that are single characters (often happens in some PDFs)
    - keep paragraph breaks where meaningful
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    # Trim right-side spaces
    lines = [ln.rstrip() for ln in text.split("\n")]

    # If we see lots of single-character lines, join them back.
    # Example pathological case: "G\na\nr\ny\n"
    single_char_lines = sum(1 for ln in lines if len(ln.strip()) == 1)
    non_empty_lines = sum(1 for ln in lines if ln.strip())
    if non_empty_lines > 0 and (single_char_lines / non_empty_lines) > 0.30:
        # Join all non-empty lines with spaces; keep blank lines as paragraph separators
        rebuilt = []
        buf = []
        for ln in lines:
            if not ln.strip():
                if buf:
                    rebuilt.append(" ".join(buf))
                    buf = []
                rebuilt.append("")  # paragraph break
            else:
                buf.append(ln.strip())
        if buf:
            rebuilt.append(" ".join(buf))
        lines = rebuilt

    # Collapse multiple blank lines to max 2
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    # Collapse excessive spaces/tabs
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)

    return normalized.strip()


def chunk_text_windowed(
    text: str,
    max_chars: int = 900,
    overlap: int = 150,
    min_chars: int = 120,
) -> list[str]:
    """
    Robust chunker:
    - normalize extracted text
    - build chunks from paragraphs/lines
    - fallback to sliding windows
    - drop tiny chunks (noise)
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >=0 and < max_chars")

    text = normalize_pdf_text(text)
    if not text:
        return []

    # First try paragraph split
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If paragraph split yields too many tiny pieces, use line-merge strategy
    if len(paras) > 50:
        paras = _merge_lines_into_blocks(text)

    chunks: list[str] = []
    buf = ""

    for p in paras:
        if len(p) < 2:
            continue
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= max_chars:
            buf = f"{buf}\n\n{p}"
        else:
            if len(buf.strip()) >= min_chars:
                chunks.append(buf.strip())
            buf = p

    if buf and len(buf.strip()) >= min_chars:
        chunks.append(buf.strip())

    # Window any oversized chunks
    final: list[str] = []
    for c in chunks:
        if len(c) <= int(max_chars * 1.3):
            final.append(c)
        else:
            final.extend(_sliding_windows(c, max_chars=max_chars, overlap=overlap, min_chars=min_chars))

    # Final filter
    final = [c for c in final if len(c.strip()) >= min_chars]
    return final


def _merge_lines_into_blocks(text: str, target_block_chars: int = 700) -> list[str]:
    """
    Merge lines into rough blocks when paragraph boundaries are unreliable.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    blocks: list[str] = []
    buf = ""

    for ln in lines:
        # Avoid gluing headings too tightly: keep headings as separate starters
        if buf and (len(buf) + 1 + len(ln) > target_block_chars):
            blocks.append(buf.strip())
            buf = ln
        else:
            buf = f"{buf} {ln}".strip() if buf else ln

    if buf:
        blocks.append(buf.strip())

    # Create pseudo-paragraphs
    return blocks


def _sliding_windows(text: str, max_chars: int, overlap: int, min_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap

    return chunks