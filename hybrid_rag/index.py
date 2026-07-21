"""Indexing: the write path into the hybrid store.

Embeds chunks, drops near-duplicates, and upserts the rest. One store holds both
the dense vector and the BM25 text per chunk. Stable chunk ids make re-indexing
an upsert.
"""

import math
from dataclasses import asdict, dataclass

from hybrid_rag.config import settings
from hybrid_rag.embeddings import embed_texts
from hybrid_rag.models import Chunk
from hybrid_rag.stores import HybridStore, build_store


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
    """Writes chunks into the hybrid store (dense vector + BM25 text per chunk)."""

    def __init__(
        self,
        store: HybridStore | None = None,
        embed_fn=embed_texts,
        dedup_threshold: float | None = None,
        index: str | None = None,
    ) -> None:
        self._store = store or build_store(index=index)
        self._embed_fn = embed_fn
        self._dedup_threshold = (
            settings.dedup_threshold if dedup_threshold is None else dedup_threshold
        )

    def add(self, chunks: list[Chunk]) -> AddResult:
        """Embed chunks, drop near-duplicates, upsert the rest into the store."""
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

        return AddResult(added=[_chunk_id(c) for c in kept], skipped=skipped)

    def _is_duplicate(
        self, chunk: Chunk, embedding: list[float], batch_embeddings: list[list[float]]
    ) -> bool:
        """Near-duplicate if cosine to a batch-mate or stored chunk tops the threshold.

        A stored hit with this chunk's own id is a re-index, not a duplicate.
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
