"""Document ingestion: turn an uploaded file into searchable chunks.

Downloads the stored bytes, then parses -> chunks -> embeds -> indexes into the
hybrid store, stamping ``document_id`` on every chunk. Updates the document's
status as it goes (``processing`` -> ``indexed`` / ``failed``).

Runs as a FastAPI ``BackgroundTask`` after the upload is confirmed. The parse/
embed/index work is synchronous and CPU/IO-heavy, so it runs in a thread to keep
the event loop free. A durable worker (Temporal) replaces this trigger later.
"""

import asyncio
import logging

from app.config import settings
from app.models.document import DocumentStatus
from app.rag.chunking import chunk as chunk_segments
from app.rag.contextual import contextualize
from app.rag.index import Index
from app.rag.loaders import load_bytes
from app.repositories.document_repo import DocumentRepository
from app.storage import StorageBackend

logger = logging.getLogger("hybrid_rag")

# Structure-aware chunking for every format: it uses (and prepends) headings for
# md/html, and degrades to recursive splitting that respects page/section
# boundaries for pdf/txt. A format->strategy switch is only worth adding when a
# format needs a different strategy (e.g. code -> AST), which no loader does yet.
_STRATEGY = "header"


class IngestService:
    def __init__(
        self,
        storage: StorageBackend,
        repo: DocumentRepository,
        index: Index | None = None,
    ) -> None:
        self._storage = storage
        self._repo = repo
        # Built lazily on first use (inside the worker thread) so OpenSearch being
        # unreachable never breaks the upload path - only the ingestion fails.
        self._index = index

    async def run(self, document_id: str) -> None:
        """Ingest one uploaded document into the search store.

        Idempotent: only acts on a document that is ``uploaded``, so a re-triggered
        task (already processing/indexed) is a no-op.
        """
        document = await self._repo.get(document_id)
        if document is None or document.status != DocumentStatus.uploaded:
            return

        await self._repo.set_status(document_id, DocumentStatus.processing)
        try:
            chunk_count = await asyncio.to_thread(
                self._index_document,
                document.storage_key,
                document.filename,
                document_id,
            )
        except Exception as exc:
            logger.exception("ingest failed for %s", document_id)
            await self._repo.set_status(
                document_id, DocumentStatus.failed, error=str(exc)[:500]
            )
            return

        await self._repo.set_status(
            document_id, DocumentStatus.indexed, chunk_count=chunk_count
        )

    def _index_document(self, storage_key: str, filename: str, document_id: str) -> int:
        """Blocking pipeline: download -> parse -> chunk -> index. Returns count."""
        data = self._storage.get(storage_key)
        segments = load_bytes(data, filename)
        chunks = chunk_segments(segments, strategy=_STRATEGY)
        if settings.contextual_retrieval:
            document = "\n\n".join(seg.text for seg in segments)
            chunks = contextualize(chunks, document)
        for c in chunks:
            c.document_id = document_id
        result = self._get_index().add(chunks)
        return len(result.added)

    def _get_index(self) -> Index:
        if self._index is None:
            self._index = Index()
        return self._index
