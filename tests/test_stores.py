"""Tests for the vector-store abstraction.

These stay offline: the factory is checked by name only, and a fake in-memory
store proves the ``Index`` depends on the ``VectorStore`` port, not on ChromaDB.
"""

import pytest

from hybrid_rag.index import Index
from hybrid_rag.models import Chunk
from hybrid_rag.sparse import Bm25Store
from hybrid_rag.stores import ChromaStore, QueryHit, VectorStore, build_store


def test_build_store_uses_configured_backend(monkeypatch):
    from hybrid_rag import config

    monkeypatch.setattr(config.settings, "vector_backend", "chroma")
    assert isinstance(build_store(), ChromaStore)


def test_build_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown vector backend"):
        build_store("cassandra")


class FakeStore:
    """Minimal in-memory VectorStore, no external service."""

    def __init__(self):
        self.records: dict[str, tuple] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            self.records[i] = (e, d, m)

    def query(self, embedding, k=10):
        items = list(self.records.items())[:k]
        return [QueryHit(i, d, m, 0.0) for i, (e, d, m) in items]

    def count(self):
        return len(self.records)


def test_index_works_against_any_vectorstore(tmp_path):
    store = FakeStore()
    assert isinstance(store, VectorStore)  # structural check via runtime Protocol

    idx = Index(
        store=store,
        sparse=Bm25Store(path=str(tmp_path / "bm25.json")),
        embed_fn=lambda texts: [[float(len(t))] for t in texts],
    )
    idx.add([Chunk("a body", "doc.md", 0, "fixed", 6)])

    assert idx.count == 1
    assert store.records["doc.md:fixed:0"][1] == "a body"
    assert idx._sparse.count() == 1
