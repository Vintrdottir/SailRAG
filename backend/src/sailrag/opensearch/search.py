from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    doc_id: str
    page_number: int
    tags: list[str]
    text: str
    score: float
    source: str  # "bm25" or "knn" or "hybrid"


async def bm25_search(
    opensearch_url: str,
    index_name: str,
    query: str,
    k: int = 10,
    timeout_s: float = 20.0,
) -> list[SearchHit]:
    body = {
        "size": k,
        "_source": ["chunk_id", "doc_id", "page_number", "tags", "text"],
        "query": {"match": {"text": {"query": query}}},
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{opensearch_url}/{index_name}/_search", json=body)
        r.raise_for_status()
        data = r.json()

    hits = []
    for h in data.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append(
            SearchHit(
                chunk_id=src.get("chunk_id", h.get("_id")),
                doc_id=src.get("doc_id", ""),
                page_number=int(src.get("page_number", 0) or 0),
                tags=list(src.get("tags") or []),
                text=src.get("text", ""),
                score=float(h.get("_score") or 0.0),
                source="bm25",
            )
        )
    return hits


async def knn_search(
    opensearch_url: str,
    index_name: str,
    query_vector: list[float],
    k: int = 10,
    timeout_s: float = 30.0,
) -> list[SearchHit]:
    body = {
        "size": k,
        "_source": ["chunk_id", "doc_id", "page_number", "tags", "text"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        },
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{opensearch_url}/{index_name}/_search", json=body)
        r.raise_for_status()
        data = r.json()

    hits = []
    for h in data.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append(
            SearchHit(
                chunk_id=src.get("chunk_id", h.get("_id")),
                doc_id=src.get("doc_id", ""),
                page_number=int(src.get("page_number", 0) or 0),
                tags=list(src.get("tags") or []),
                text=src.get("text", ""),
                score=float(h.get("_score") or 0.0),
                source="knn",
            )
        )
    return hits


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def fuse_hits(
    bm25_hits: list[SearchHit],
    knn_hits: list[SearchHit],
    w_bm25: float = 0.5,
    w_knn: float = 0.5,
) -> list[SearchHit]:
    """
    Weighted score fusion after min-max normalization within each list.
    """
    bm25_scores = _minmax_normalize([h.score for h in bm25_hits])
    knn_scores = _minmax_normalize([h.score for h in knn_hits])

    bm25_map = {h.chunk_id: (h, bm25_scores[i]) for i, h in enumerate(bm25_hits)}
    knn_map = {h.chunk_id: (h, knn_scores[i]) for i, h in enumerate(knn_hits)}

    all_ids = set(bm25_map.keys()) | set(knn_map.keys())

    fused: list[SearchHit] = []
    for cid in all_ids:
        b = bm25_map.get(cid)
        k = knn_map.get(cid)

        # Prefer metadata/text from bm25 if present, else from knn
        base = (b[0] if b else k[0])

        b_score = b[1] if b else 0.0
        k_score = k[1] if k else 0.0
        score = w_bm25 * b_score + w_knn * k_score

        fused.append(
            SearchHit(
                chunk_id=base.chunk_id,
                doc_id=base.doc_id,
                page_number=base.page_number,
                tags=base.tags,
                text=base.text,
                score=score,
                source="hybrid",
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused


def to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "page_number": hit.page_number,
        "tags": hit.tags,
        "score": hit.score,
        "source": hit.source,
        "text": hit.text,
    }
