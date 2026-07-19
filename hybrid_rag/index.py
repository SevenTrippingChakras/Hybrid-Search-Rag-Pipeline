"""Indexing: the write path into the dense and sparse stores, kept in sync.

This module only writes. It embeds chunks, drops near-duplicates, and upserts the
survivors into a swappable dense ``VectorStore`` (:mod:`hybrid_rag.stores`) and a
``SparseStore`` (BM25, :mod:`hybrid_rag.sparse`). Both stay in sync because every
``add`` writes both, and stable chunk ids make re-indexing an upsert on each side.

Reading the stores back — dense search, sparse search, fusion, rerank — is the
retriever's job (:mod:`hybrid_rag.retrieval`); this module never searches. That
split mirrors production: indexing and retrieval are separate concerns over the
same shared stores.

Embeddings are computed here via ``embed_fn`` (injectable, so indexing runs
offline in tests) and handed to the dense store directly.

Before inserting, ``add`` drops near-duplicate chunks (Phase 1.4): a chunk whose
cosine similarity to an already-stored chunk (or to an earlier chunk in the same
batch) exceeds ``dedup_threshold`` is skipped, so retrieval never wastes context
on the same content appearing in two docs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from hybrid_rag.config import settings
from hybrid_rag.embeddings import embed_texts
from hybrid_rag.models import Chunk
from hybrid_rag.sparse import Bm25Store, SparseStore
from hybrid_rag.stores import VectorStore, build_store


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class AddResult:
    """What ``add`` did: ids actually inserted, and ids skipped as duplicates."""

    added: list[str]
    skipped: list[str]


def _chunk_id(chunk: Chunk) -> str:
    """Stable id so re-indexing the same chunk upserts instead of duplicating."""
    return f"{chunk.source}:{chunk.strategy}:{chunk.chunk_index}"


def _metadata(chunk: Chunk) -> dict:
    """Chunk provenance as store metadata (None-valued fields dropped)."""
    data = asdict(chunk)
    data.pop("text")
    return {k: v for k, v in data.items() if v is not None}


class Index:
    """Writes chunks into a dense store and a sparse store, keeping them in sync."""

    def __init__(
        self,
        store: VectorStore | None = None,
        sparse: SparseStore | None = None,
        embed_fn=embed_texts,
        dedup_threshold: float | None = None,
    ) -> None:
        self._store = store or build_store()
        self._sparse = sparse or Bm25Store()
        self._embed_fn = embed_fn
        self._dedup_threshold = (
            settings.dedup_threshold if dedup_threshold is None else dedup_threshold
        )

    def add(self, chunks: list[Chunk]) -> AddResult:
        """Embed chunks, drop near-duplicates, upsert the rest into both stores."""
        if not chunks:
            return AddResult(added=[], skipped=[])
        embeddings = self._embed_fn([c.text for c in chunks])

        kept: list[Chunk] = []
        kept_embeddings: list[list[float]] = []
        skipped: list[str] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if self._is_duplicate(chunk, embedding, kept_embeddings):
                skipped.append(_chunk_id(chunk))
                continue
            kept.append(chunk)
            kept_embeddings.append(embedding)

        if kept:
            ids = [_chunk_id(c) for c in kept]
            documents = [c.text for c in kept]
            metadatas = [_metadata(c) for c in kept]
            self._store.upsert(
                ids=ids,
                embeddings=kept_embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            self._sparse.add(ids=ids, documents=documents, metadatas=metadatas)

        return AddResult(added=[_chunk_id(c) for c in kept], skipped=skipped)

    def _is_duplicate(
        self, chunk: Chunk, embedding: list[float], batch_embeddings: list[list[float]]
    ) -> bool:
        """Near-duplicate if cosine to a batch-mate or stored chunk tops the threshold.

        A stored hit with this chunk's own id is a re-index (upsert), not a
        duplicate, so it is ignored.
        """
        if self._dedup_threshold >= 1.0:
            return False
        for other in batch_embeddings:
            if _cosine(embedding, other) > self._dedup_threshold:
                return True
        own_id = _chunk_id(chunk)
        for hit in self._store.query(embedding, k=2):
            if hit.id != own_id and hit.score > self._dedup_threshold:
                return True
        return False

    @property
    def count(self) -> int:
        return self._store.count()
