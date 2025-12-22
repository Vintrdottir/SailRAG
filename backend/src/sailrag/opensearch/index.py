from __future__ import annotations

import httpx


def build_index_body(embedding_dim: int) -> dict:
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "tags": {"type": "keyword"},
                "text": {"type": "text"},  # BM25
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },
            }
        },
    }


async def ensure_index(opensearch_url: str, index_name: str, embedding_dim: int) -> dict:
    body = build_index_body(embedding_dim)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check if exists
        r = await client.head(f"{opensearch_url}/{index_name}")
        if r.status_code == 200:
            return {"status": "exists", "index": index_name}

        # Create
        r = await client.put(f"{opensearch_url}/{index_name}", json=body)
        r.raise_for_status()
        return {"status": "created", "index": index_name}
