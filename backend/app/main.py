"""FastAPI application entry point.

Thin composition root: creates the app and mounts the route modules. All
business logic lives in services; routes stay thin. Run locally with::

    uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Hybrid RAG API", version="0.1.0")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
