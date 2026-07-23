"""Retrieval: the read path over the hybrid store.

The ``Retriever`` depends only on the store port plus an ``embed_fn`` and shares
the store instance the ``Index`` writes to.
"""

from app.config import settings
from app.rag.embeddings import embed_texts
from app.rag.reranker import CrossEncoderReranker, Reranker
from app.rag.stores import HybridStore, QueryHit, build_store


class Retriever:
    """Reads the hybrid store's dense and sparse sides to find relevant chunks."""

    def __init__(
        self,
        store: HybridStore | None = None,
        reranker: Reranker | None = None,
        embed_fn=embed_texts,
        index: str | None = None,
    ) -> None:
        self._store = store or build_store(index=index)
        self._reranker = reranker or CrossEncoderReranker()
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
        return self._store.sparse_query(query, k=k)

    def hybrid_search(self, query: str, k: int = 10) -> list[QueryHit]:
        """OpenSearch native hybrid: BM25 + k-NN fused server-side by RRF.

        One request; the engine runs both sub-queries and combines them with
        Reciprocal Rank Fusion. Semantic recall plus exact keyword matching in a
        single ranked list.
        """
        embedding = self._embed_fn([query])[0]
        return self._store.hybrid_query(query, embedding, k=k)

    def warmup(self) -> None:
        """Pre-load the reranker model so the first search doesn't pay for it."""
        self._reranker.warmup()

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
        """The full read pipeline: native hybrid fusion then cross-encoder rerank."""
        candidates = self.hybrid_search(query, k=candidate_k)
        return self.rerank(query, candidates, top_k=top_k)
