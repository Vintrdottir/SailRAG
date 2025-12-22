from __future__ import annotations

import json
import httpx


async def bulk_index(opensearch_url: str, index_name: str, docs: list[dict]) -> dict:
    """
    Bulk index documents. Each doc must contain an 'id' key used as _id.
    """
    lines = []
    for d in docs:
        doc_id = d.pop("id")
        lines.append(json.dumps({"index": {"_index": index_name, "_id": doc_id}}))
        lines.append(json.dumps(d))

    payload = "\n".join(lines) + "\n"

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{opensearch_url}/_bulk",
            content=payload,
            headers={"Content-Type": "application/x-ndjson"},
        )
        r.raise_for_status()
        data = r.json()
        return {"errors": data.get("errors", False), "items": len(data.get("items", []))}
