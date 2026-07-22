"""Dependency providers for FastAPI's ``Depends``.

Routes declare what they need (a ``DocumentService``); these functions build it.
The storage client is cached so boto3 isn't re-initialized per request; the
repository is stateless over the shared Mongo client, so a fresh instance is fine.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.repositories.document_repo import DocumentRepository
from app.services.document_service import DocumentService
from app.services.ingest_service import IngestService
from app.storage import StorageBackend, build_storage


@lru_cache
def get_storage() -> StorageBackend:
    return build_storage()


def get_document_repo() -> DocumentRepository:
    return DocumentRepository()


def get_document_service(
    storage: Annotated[StorageBackend, Depends(get_storage)],
    repo: Annotated[DocumentRepository, Depends(get_document_repo)],
) -> DocumentService:
    return DocumentService(storage=storage, repo=repo)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_ingest_service(
    storage: Annotated[StorageBackend, Depends(get_storage)],
    repo: Annotated[DocumentRepository, Depends(get_document_repo)],
) -> IngestService:
    return IngestService(storage=storage, repo=repo)


IngestServiceDep = Annotated[IngestService, Depends(get_ingest_service)]
