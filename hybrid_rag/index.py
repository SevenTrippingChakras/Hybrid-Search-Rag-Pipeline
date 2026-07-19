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
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from hybrid_rag.embeddings import embed_texts
from hybrid_rag.models import Chunk
from hybrid_rag.stores import VectorStore, build_store

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
    ) -> None:
        self._store = store or build_store()
        self._embed_fn = embed_fn
        self._sparse_path = Path(sparse_path)
        self._corpus: dict[str, str] = self._load_corpus()
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_sparse()

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into the dense store, then update sparse."""
        if not chunks:
            return
        embeddings = self._embed_fn([c.text for c in chunks])
        self._store.upsert(
            ids=[_chunk_id(c) for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )
        for c in chunks:
            self._corpus[_chunk_id(c)] = c.text
        self._save_corpus()
        self._rebuild_sparse()

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
