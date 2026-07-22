"""The document record: what we persist about an uploaded file.

This is the domain/DB shape, distinct from the API request/response schemas in
the route module. Embeddings never live here - only metadata and lifecycle
status. The raw bytes live in object storage under ``storage_key``.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document.

    ``pending``  -> presigned URL issued, waiting for the client PUT.
    ``uploaded`` -> bytes confirmed in storage (ingestion wires in later here).
    ``processing``/``indexed``/``failed`` -> reserved for the ingestion phase.
    """

    pending = "pending"
    uploaded = "uploaded"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


def _now() -> datetime:
    return datetime.now(UTC)


class Document(BaseModel):
    """One uploaded document's metadata record (the persisted shape)."""

    document_id: str
    filename: str
    content_type: str
    size: int
    storage_key: str
    status: DocumentStatus
    chunk_count: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class InitiateUploadRequest(BaseModel):
    """Body of ``POST /documents``: metadata only, no bytes."""

    filename: str
    content_type: str
    size: int


class InitiateUploadResponse(BaseModel):
    """Reply to ``POST /documents``: the record id and where to PUT the file."""

    document_id: str
    upload_url: str
