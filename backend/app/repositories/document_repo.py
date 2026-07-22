"""Mongo persistence for document metadata.

The only layer that touches the database. Services depend on this class, not on
pymongo, so the store could be swapped without touching business logic. The
document_id doubles as Mongo's ``_id`` so lookups and upserts are by primary key.
"""

from datetime import UTC, datetime

from app.db import get_db
from app.models.document import Document, DocumentStatus


class DocumentRepository:
    """CRUD for document records in the ``documents`` collection."""

    def __init__(self) -> None:
        self._collection = get_db()["documents"]

    async def create(self, document: Document) -> None:
        doc = document.model_dump()
        doc["_id"] = document.document_id
        await self._collection.insert_one(doc)

    async def get(self, document_id: str) -> Document | None:
        doc = await self._collection.find_one({"_id": document_id})
        return Document(**doc) if doc else None

    async def list(self) -> list[Document]:
        cursor = self._collection.find().sort("created_at", -1)
        return [Document(**doc) async for doc in cursor]

    async def set_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        fields: dict = {
            "status": status.value,
            "updated_at": datetime.now(UTC),
        }
        if chunk_count is not None:
            fields["chunk_count"] = chunk_count
        if error is not None:
            fields["error"] = error
        await self._collection.update_one({"_id": document_id}, {"$set": fields})

    async def delete(self, document_id: str) -> None:
        await self._collection.delete_one({"_id": document_id})
