"""Dense + sparse index over chunks, kept in sync.

Pairs a swappable dense ``VectorStore`` (see :mod:`hybrid_rag.stores`) with a
BM25 keyword index. Dense vectors go to whichever backend ``VECTOR_BACKEND``
selects; the sparse index is maintained here from a small local corpus sidecar,
so BM25 behaves identically whether the dense backend is Chroma, Pinecone, or
Milvus (Pinecone, for one, cannot enumerate all vectors to rebuild from).

The two indexes stay in sync because every ``add`` writes both: the dense store
and the sparse corpus. Stable chunk ids make re-indexing an upsert on both sides.
Embeddings are computed here via ``embed_fn`` (injectable, so the index builds
offline) and handed to the store directly.

Before inserting, ``add`` drops near-duplicate chunks (Phase 1.4): a chunk whose
cosine similarity to an already-stored chunk (or to an earlier chunk in the same
batch) exceeds ``dedup_threshold`` is skipped, so the retriever never wastes
context on the same content appearing in two docs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from hybrid_rag.config import settings
from hybrid_rag.embeddings import embed_texts
from hybrid_rag.models import Chunk
from hybrid_rag.stores import VectorStore, build_store

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
    """A swappable dense store paired with a local BM25 sparse index."""

    def __init__(
        self,
        store: VectorStore | None = None,
        sparse_path: str = "data/index/bm25.json",
        embed_fn=embed_texts,
        dedup_threshold: float | None = None,
    ) -> None:
        self._store = store or build_store()
        self._embed_fn = embed_fn
        self._dedup_threshold = (
            settings.dedup_threshold if dedup_threshold is None else dedup_threshold
        )
        self._sparse_path = Path(sparse_path)
        self._corpus: dict[str, str] = self._load_corpus()
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_sparse()

    def add(self, chunks: list[Chunk]) -> AddResult:
        """Embed chunks, drop near-duplicates, upsert the rest, update sparse."""
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
            self._store.upsert(
                ids=[_chunk_id(c) for c in kept],
                embeddings=kept_embeddings,
                documents=[c.text for c in kept],
                metadatas=[_metadata(c) for c in kept],
            )
            for c in kept:
                self._corpus[_chunk_id(c)] = c.text
            self._save_corpus()
            self._rebuild_sparse()

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

    def _rebuild_sparse(self) -> None:
        """Build the BM25 index from the local corpus of chunk texts."""
        self._ids = list(self._corpus)
        tokens = [_tokenize(self._corpus[i]) for i in self._ids]
        self._bm25 = BM25Okapi(tokens) if tokens else None

    def _load_corpus(self) -> dict[str, str]:
        if self._sparse_path.exists():
            return json.loads(self._sparse_path.read_text(encoding="utf-8"))
        return {}

    def _save_corpus(self) -> None:
        self._sparse_path.parent.mkdir(parents=True, exist_ok=True)
        self._sparse_path.write_text(
            json.dumps(self._corpus, ensure_ascii=False), encoding="utf-8"
        )
