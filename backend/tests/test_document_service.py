"""Tests for DocumentService upload validation.

The service must refuse file types the ingestion loaders cannot parse, before
any record is written or a presigned URL is minted - so a doomed upload fails
fast instead of at ingestion time.
"""

import asyncio

import pytest

from app.core.errors import UnsupportedFileType
from app.services.document_service import DocumentService
from tests.support import FakeStorage


class FakeRepo:
    """In-memory document records; enough for the upload path."""

    def __init__(self):
        self.records = {}

    async def create(self, document):
        self.records[document.document_id] = document


def _service():
    return DocumentService(storage=FakeStorage(), repo=FakeRepo())


def test_initiate_upload_rejects_unsupported_type():
    service = _service()
    with pytest.raises(UnsupportedFileType):
        asyncio.run(
            service.initiate_upload(
                filename="archive.zip", content_type="application/zip", size=10
            )
        )
    assert service._repo.records == {}


def test_initiate_upload_accepts_supported_type():
    service = _service()
    document, upload_url = asyncio.run(
        service.initiate_upload(
            filename="notes.md", content_type="text/markdown", size=10
        )
    )
    assert document.filename == "notes.md"
    assert upload_url.startswith("https://")
    assert service._repo.records == {document.document_id: document}
