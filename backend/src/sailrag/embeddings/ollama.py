from __future__ import annotations

from pydantic import BaseModel
import httpx


class EmbedResponse(BaseModel):
    embedding: list[float]


async def embed_text_ollama(
    ollama_url: str,
    model: str,
    text: str,
    timeout_s: float = 60.0,
) -> list[float]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        r.raise_for_status()
        data = EmbedResponse.model_validate(r.json())
        return data.embedding