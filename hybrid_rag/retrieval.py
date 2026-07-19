"""Retrieval: the read path over the dense and sparse stores.

Separate from indexing (:mod:`hybrid_rag.index`) on purpose — indexing writes the
stores, retrieval reads them. The ``Retriever`` depends only on the store ports
plus an ``embed_fn``, so it works across every dense backend and, once Phase 2.5
lands, over OpenSearch with almost no change. The same store instances are shared
with the ``Index``; by default both point at the same persisted paths.

Phase 2 builds this up in points, each an additive method over the same two
stores: dense search (2.1), sparse BM25 search (2.2), RRF fusion of the two
(2.3), and a reranker (2.4).
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
        """Dense retrieval (Phase 2.1): embed the query, return top-k by cosine.

        The query is embedded with the same ``embed_fn`` used at index time, then
        handed to the dense store, which ranks chunks by cosine similarity. This
        finds semantically related passages even when they share no keywords — the
        sparse (BM25) half in Phase 2.2 covers the exact-term case.
        """
        embedding = self._embed_fn([query])[0]
        return self._store.query(embedding, k=k)
