"""FastAPI application entry point.

Thin composition root: creates the app and mounts the route modules. All
business logic lives in services; routes stay thin. Run locally with::

    uv run uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import db
from app.routes import documents


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Mongo connection on startup, close it on shutdown."""
    await db.connect()
    yield
    await db.close()


app = FastAPI(title="Hybrid RAG API", version="0.1.0", lifespan=lifespan)

app.include_router(documents.router)


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
    for name, ping in (("mongo", _ping_mongo),):
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
