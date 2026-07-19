"""Tests for the cross-encoder reranker.

A stub scorer is injected in place of the model so the suite runs offline.
"""

from hybrid_rag.reranker import CrossEncoderReranker
from hybrid_rag.stores import QueryHit


class _StubModel:
    """Stands in for the loaded CrossEncoder: scores by query-token overlap."""

    def predict(self, pairs):
        scores = []
        for query, doc in pairs:
            q = set(query.lower().split())
            overlap = sum(1 for w in doc.lower().split() if w in q)
            scores.append(float(overlap))
        return scores


def _reranker():
    r = CrossEncoderReranker()
    r._model = _StubModel()  # skip the real model load
    return r


def _hit(id_, doc, score=0.0):
    return QueryHit(id_, doc, {"source": "doc.md"}, score)


def test_rerank_orders_by_cross_encoder_score():
    reranker = _reranker()
    hits = [
        _hit("a", "the cat sat on the mat", score=0.9),
        _hit("b", "restart the engine to clear the error", score=0.1),
    ]

    reranked = reranker.rerank("engine error", hits, top_k=2)

    assert reranked[0].id == "b"  # more query-relevant despite lower input score
    assert reranked[0].score > reranked[1].score


def test_rerank_respects_top_k():
    reranker = _reranker()
    hits = [_hit(str(n), f"unique words number {n}") for n in range(5)]

    assert len(reranker.rerank("words", hits, top_k=3)) == 3


def test_rerank_preserves_metadata():
    reranker = _reranker()
    hit = reranker.rerank("engine", [_hit("a", "engine oil")], top_k=1)[0]

    assert hit.metadata["source"] == "doc.md"


def test_rerank_empty_returns_nothing():
    assert _reranker().rerank("anything", [], top_k=5) == []
