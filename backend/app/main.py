"""FastAPI application entry point.

Thin composition root: creates the app and mounts the route modules. All
business logic lives in services; routes stay thin. Run locally with::

    uv run uvicorn app.main:app --reload
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db
from app.config import settings
from app.core.errors import register_error_handlers
from app.deps import get_pipeline
from app.routes import ask, documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open Mongo and pre-load the reranker on startup; close Mongo on shutdown.

    Warming the cross-encoder here downloads (first ever run) and loads it into
    memory during startup, so the first question doesn't pay that cost.
    """
    await db.connect()
    await asyncio.to_thread(get_pipeline().warmup)
    yield
    await db.close()


app = FastAPI(title="Hybrid RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_error_handlers(app)
app.include_router(documents.router)
app.include_router(ask.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe: the process is up. Touches no dependencies."""
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
async def ready() -> JSONResponse:
    """Readiness probe: pings each dependency; 503 if any is unreachable.

    Orchestrators use this to decide whether to route traffic here, distinct
    from ``/health`` liveness which only says the process is alive.
    """
    checks: dict[str, str] = {}
    for name, ping in (("mongo", _ping_mongo), ("opensearch", _ping_opensearch)):
        try:
            await ping()
            checks[name] = "ok"
        except Exception:
            checks[name] = "error"

    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "error", "checks": checks},
    )


async def _ping_mongo() -> None:
    await db.get_db().command("ping")


async def _ping_opensearch() -> None:
    from opensearchpy import OpenSearch

    def _ping() -> None:
        client = OpenSearch(hosts=[settings.opensearch_host])
        if not client.ping():
            raise RuntimeError("opensearch ping failed")

    await asyncio.to_thread(_ping)
