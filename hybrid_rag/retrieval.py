"""Retrieval: the read path over the dense and sparse stores.

The ``Retriever`` depends only on the store ports plus an ``embed_fn`` and shares
the store instances the ``Index`` writes to.
"""

from hybrid_rag.config import settings
from hybrid_rag.embeddings import embed_texts
from hybrid_rag.fusion import reciprocal_rank_fusion
from hybrid_rag.reranker import CrossEncoderReranker, Reranker
from hybrid_rag.sparse import Bm25Store, SparseStore
from hybrid_rag.stores import QueryHit, VectorStore, build_store


class Retriever:
    """Reads the dense and sparse stores to find chunks relevant to a query."""

    def __init__(
        self,
        store: VectorStore | None = None,
        sparse: SparseStore | None = None,
        reranker: Reranker | None = None,
        embed_fn=embed_texts,
        dense_weight: float = settings.dense_weight,
        sparse_weight: float = settings.sparse_weight,
        rrf_k: int = settings.rrf_k,
    ) -> None:
        self._store = store or build_store()
        self._sparse = sparse or Bm25Store()
        self._reranker = reranker or CrossEncoderReranker()
        self._embed_fn = embed_fn
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._rrf_k = rrf_k

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

    def hybrid_search(
        self, query: str, k: int = 10, candidate_k: int = 20
    ) -> list[QueryHit]:
        """Fuse dense and sparse results with weighted Reciprocal Rank Fusion.

        Pulls ``candidate_k`` hits from each retriever, merges them by rank into a
        single list (dense/sparse weights and RRF constant set at construction),
        and returns the top ``k``. Getting the best of both — semantic recall plus
        exact keyword matching — in one ranked list.
        """
        dense = self.dense_search(query, k=candidate_k)
        sparse = self.sparse_search(query, k=candidate_k)
        return reciprocal_rank_fusion(
            [(dense, self._dense_weight), (sparse, self._sparse_weight)],
            k=self._rrf_k,
            top_k=k,
        )

    def rerank(
        self, query: str, hits: list[QueryHit], top_k: int = settings.rerank_top_k
    ) -> list[QueryHit]:
        """Re-score fused candidates with the cross-encoder, keep the best top_k."""
        return self._reranker.rerank(query, hits, top_k=top_k)

    def search(
        self,
        query: str,
        top_k: int = settings.rerank_top_k,
        candidate_k: int = 20,
    ) -> list[QueryHit]:
        """The full read pipeline: hybrid fusion then cross-encoder rerank."""
        candidates = self.hybrid_search(query, k=candidate_k, candidate_k=candidate_k)
        return self.rerank(query, candidates, top_k=top_k)
