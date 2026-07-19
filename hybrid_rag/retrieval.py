"""Retrieval: the read path over the dense and sparse stores.

The ``Retriever`` depends only on the store ports plus an ``embed_fn`` and shares
the store instances the ``Index`` writes to.
"""

from __future__ import annotations

from hybrid_rag.embeddings import embed_texts
from hybrid_rag.sparse import Bm25Store, SparseStore
from hybrid_rag.stores import QueryHit, VectorStore, build_store


class Retriever:
    """Reads the dense and sparse stores to find chunks relevant to a query."""

    def __init__(
        self,
        store: VectorStore | None = None,
        sparse: SparseStore | None = None,
        embed_fn=embed_texts,
    ) -> None:
        self._store = store or build_store()
        self._sparse = sparse or Bm25Store()
        self._embed_fn = embed_fn

    def dense_search(self, query: str, k: int = 10) -> list[QueryHit]:
        """Embed the query and return the dense store's top-k chunks by cosine."""
        embedding = self._embed_fn([query])[0]
        return self._store.query(embedding, k=k)

    def sparse_search(self, query: str, k: int = 10) -> list[QueryHit]:
        """Return the BM25 store's top-k chunks by keyword-match score.

        Catches exact terms — function names, config keys, error codes — that
        dense search can miss.
        """
        return self._sparse.query(query, k=k)
