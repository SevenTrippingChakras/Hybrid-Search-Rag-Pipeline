"""Reranker: a precision second pass that re-scores the fused candidates.

``Reranker`` is the port, ``CrossEncoderReranker`` the cross-encoder adapter.
"""

from typing import Protocol, runtime_checkable

from app.config import settings
from app.rag.stores import QueryHit


@runtime_checkable
class Reranker(Protocol):
    """The reranker port: re-score candidates and keep the best ``top_k``."""

    def rerank(
        self, query: str, hits: list[QueryHit], top_k: int = 5
    ) -> list[QueryHit]: ...

    def warmup(self) -> None:
        """Load the model ahead of the first request (no-op if already loaded)."""


class CrossEncoderReranker:
    """A local cross-encoder (BGE) scoring query-chunk pairs, loaded lazily."""

    def __init__(self, model: str = settings.reranker_model) -> None:
        self._model_name = model
        self._model = None  # loaded lazily on first rerank

    def rerank(
        self, query: str, hits: list[QueryHit], top_k: int = 5
    ) -> list[QueryHit]:
        """Score each hit against the query and return the top_k re-ranked."""
        if not hits:
            return []
        scores = self._load().predict([(query, hit.document) for hit in hits])
        reranked = [
            QueryHit(hit.id, hit.document, hit.metadata, float(score))
            for hit, score in zip(hits, scores, strict=True)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k]

    def warmup(self) -> None:
        """Download (first ever) and load the model so the first rerank is fast."""
        self._load()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model
