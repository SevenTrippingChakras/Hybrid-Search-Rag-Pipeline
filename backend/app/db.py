"""MongoDB connection: one async client, opened on startup, closed on shutdown.

The client is created inside ``connect()`` (called from the app lifespan) rather
than at import time, so it binds to the running event loop. Repositories call
``get_db()``; they never construct a client themselves.
"""

from typing import Any

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.config import settings

_client: AsyncMongoClient[dict[str, Any]] | None = None
_db: AsyncDatabase[dict[str, Any]] | None = None


def get_db() -> AsyncDatabase[dict[str, Any]]:
    """Return the active database. ``connect()`` must have run at startup."""
    if _db is None:
        raise RuntimeError("Database not initialized. Call connect() on startup.")
    return _db


async def connect() -> None:
    """Open the client and verify the connection with a ping.

    ``tz_aware`` makes Mongo return timezone-aware UTC datetimes, so timestamps
    round-trip consistently instead of coming back naive.
    """
    global _client, _db
    # serverSelectionTimeoutMS bounds how long an operation waits for a reachable
    # server, so a down Mongo fails fast (startup and the /ready probe) instead of
    # hanging on the 30s default.
    _client = AsyncMongoClient(
        settings.mongo_uri, tz_aware=True, serverSelectionTimeoutMS=5000
    )
    _db = _client[settings.mongo_db]
    await _client.admin.command("ping")


async def close() -> None:
    """Close the client on shutdown."""
    if _client is not None:
        await _client.close()
