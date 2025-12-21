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
