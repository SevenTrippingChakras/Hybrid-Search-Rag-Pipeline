"""Document lifecycle: the presigned-upload flow and CRUD.

Owns the business rules; routes call these methods and stay thin. Collaborators
(storage, repository) are injected so the service is testable with fakes. This
service stops at "bytes are in storage" - triggering ingestion (parse -> chunk
-> embed -> index) is a separate concern wired in a later phase.
"""

import uuid

from app.models.document import Document, DocumentStatus
from app.repositories.document_repo import DocumentRepository
from app.storage import StorageBackend


class DocumentNotFound(Exception):
    """No document record for the given id."""


class UploadNotFound(Exception):
    """Client called complete but the object is not in storage."""


class DocumentService:
    def __init__(self, storage: StorageBackend, repo: DocumentRepository) -> None:
        self._storage = storage
        self._repo = repo

    async def initiate_upload(
        self, filename: str, content_type: str, size: int
    ) -> tuple[Document, str]:
        """Create a pending record and return it with a presigned PUT URL.

        The document_id prefixes the storage key so two files with the same name
        never collide.
        """
        document_id = uuid.uuid4().hex
        storage_key = f"documents/{document_id}/{filename}"
        document = Document(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            size=size,
            storage_key=storage_key,
            status=DocumentStatus.pending,
        )
        await self._repo.create(document)
        upload_url = self._storage.presigned_put_url(
            storage_key, content_type=content_type
        )
        return document, upload_url

    async def complete_upload(self, document_id: str) -> Document:
        """Confirm the client's upload landed, then flip status to uploaded.

        Never trusts the client: verifies the object exists in storage first.
        Idempotent - calling twice returns the record without re-transitioning.
        """
        document = await self._repo.get(document_id)
        if document is None:
            raise DocumentNotFound(document_id)
        if document.status != DocumentStatus.pending:
            return document
        if not self._storage.exists(document.storage_key):
            raise UploadNotFound(document_id)
        await self._repo.set_status(document_id, DocumentStatus.uploaded)
        document.status = DocumentStatus.uploaded
        return document

    async def list_documents(self) -> list[Document]:
        return await self._repo.list()

    async def get_document(self, document_id: str) -> Document:
        document = await self._repo.get(document_id)
        if document is None:
            raise DocumentNotFound(document_id)
        return document

    async def delete_document(self, document_id: str) -> None:
        """Remove the record and its object from storage."""
        document = await self._repo.get(document_id)
        if document is None:
            raise DocumentNotFound(document_id)
        self._storage.delete(document.storage_key)
        await self._repo.delete(document_id)
