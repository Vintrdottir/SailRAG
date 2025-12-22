from __future__ import annotations

import httpx


async def generate_ollama(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout_s: float = 120.0,
) -> str:
    """
    Simple non-streaming generation.
    """
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                },
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
