"""Graceful "I don't know" handling (Phase 3.4).

A RAG system that always answers will hallucinate when retrieval is weak. The
``AbstentionGate`` sits before generation: if retrieval confidence (the Phase 3.3
signal, mean rerank score) is below a threshold, it returns a structured
``NoAnswer`` describing what was found, that it could not answer confidently, and
which source documents are worth checking by hand -- more useful than a
fabricated answer, and a signal of production maturity.

No LLM call: the decision is a pure computation over the reranked hits.
"""

from hybrid_rag.confidence import retrieval_confidence
from hybrid_rag.config import settings
from hybrid_rag.models import NoAnswer
from hybrid_rag.stores import QueryHit


class AbstentionGate:
    """Decides whether retrieval is strong enough to answer at all."""

    def __init__(self, threshold: float | None = None) -> None:
        self._threshold = (
            threshold if threshold is not None else settings.abstain_threshold
        )

    def check(self, query: str, hits: list[QueryHit]) -> NoAnswer | None:
        """Return a ``NoAnswer`` if confidence is too low, else ``None`` to proceed."""
        confidence = retrieval_confidence(hits)
        if confidence >= self._threshold:
            return None
        return NoAnswer(
            query=query,
            retrieval_confidence=confidence,
            message=(
                "The retrieved documents do not clearly answer this question, so "
                "no grounded answer was generated. The sources below were the "
                "closest matches and may be worth checking manually."
            ),
            found=self._describe(hits),
            suggested_sources=self._sources(hits),
        )

    @staticmethod
    def _describe(hits: list[QueryHit]) -> list[str]:
        """Short 'source -- heading' labels for the closest matches found."""
        labels = []
        for hit in hits:
            source = hit.metadata.get("source", "unknown")
            heading = hit.metadata.get("heading")
            labels.append(f"{source} — {heading}" if heading else source)
        return labels

    @staticmethod
    def _sources(hits: list[QueryHit]) -> list[str]:
        """Distinct source documents worth checking, in retrieval order."""
        seen: list[str] = []
        for hit in hits:
            source = hit.metadata.get("source", "unknown")
            if source not in seen:
                seen.append(source)
        return seen
