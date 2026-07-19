"""Swappable vector-store backends behind one interface.

``VectorStore`` is the port; ``ChromaStore``, ``PineconeStore``, ``MilvusStore``
are the adapters. ``build_store`` selects one from the ``VECTOR_BACKEND`` env var,
so the backend is a deployment choice, not a code change: Chroma (local, zero
setup) for dev and tests, Pinecone or Milvus for production.

Embeddings are computed upstream and handed in, so every backend stores the same
``text-embedding-3-small`` vectors. Cosine similarity is used throughout; each
adapter's ``query`` returns a ``score`` where higher means more similar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hybrid_rag.config import settings

EMBED_DIM = 1536  # text-embedding-3-small


@dataclass
class QueryHit:
    """One search result: the stored chunk, its metadata, and a similarity score."""

    id: str
    document: str
    metadata: dict
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """The dense-store port. Adapters compute nothing; they store and search."""

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(self, embedding: list[float], k: int = 10) -> list[QueryHit]: ...

    def count(self) -> int: ...


class ChromaStore:
    """Embedded ChromaDB: persists to a local folder, no server or credentials."""

    def __init__(self, path: str = "data/index", collection: str = "chunks") -> None:
        import chromadb

        client = chromadb.PersistentClient(path=path)
        self._collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding, k: int = 10) -> list[QueryHit]:
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res["ids"] or [[]])[0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]
        return [
            QueryHit(id_, doc, dict(meta), 1.0 - dist)
            for id_, doc, meta, dist in zip(ids, docs, metas, dists, strict=True)
        ]

    def count(self) -> int:
        return self._collection.count()


def build_store(backend: str | None = None) -> VectorStore:
    """Construct the store named by ``VECTOR_BACKEND`` (default ``chroma``)."""
    backend = (backend or settings.vector_backend).lower()
    if backend == "chroma":
        return ChromaStore(path=settings.chroma_path)
    if backend == "pinecone":
        from hybrid_rag.stores_cloud import PineconeStore

        return PineconeStore()
    if backend == "milvus":
        from hybrid_rag.stores_cloud import MilvusStore

        return MilvusStore()
    raise ValueError(f"Unknown vector backend: {backend!r}")
