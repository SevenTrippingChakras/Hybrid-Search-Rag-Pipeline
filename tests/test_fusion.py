"""Tests for Reciprocal Rank Fusion — the pure merge, no stores involved."""

from hybrid_rag.fusion import reciprocal_rank_fusion
from hybrid_rag.stores import QueryHit


def _hit(id_, score=0.0):
    return QueryHit(id_, f"doc {id_}", {"source": "x"}, score)


def test_agreement_ranks_shared_top_first():
    """A doc ranked highly by both lists beats docs in only one."""
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("a"), _hit("d"), _hit("e")]

    fused = reciprocal_rank_fusion([(dense, 1.0), (sparse, 1.0)], k=60)

    assert fused[0].id == "a"


def test_score_is_rrf_not_original():
    dense = [_hit("a", score=0.99)]
    fused = reciprocal_rank_fusion([(dense, 1.0)], k=60)
    assert fused[0].score == 1.0 / (60 + 1)


def test_weights_shift_ranking():
    """The heavier-weighted list's top hit wins when the two disagree."""
    dense = [_hit("d1")]
    sparse = [_hit("s1")]

    dense_heavy = reciprocal_rank_fusion([(dense, 0.9), (sparse, 0.1)], k=60)
    sparse_heavy = reciprocal_rank_fusion([(dense, 0.1), (sparse, 0.9)], k=60)

    assert dense_heavy[0].id == "d1"
    assert sparse_heavy[0].id == "s1"


def test_top_k_truncates():
    dense = [_hit(str(n)) for n in range(5)]
    assert len(reciprocal_rank_fusion([(dense, 1.0)], top_k=3)) == 3


def test_empty_lists_return_nothing():
    assert reciprocal_rank_fusion([([], 1.0), ([], 1.0)]) == []


def test_dedupes_across_lists():
    """The same id in both lists appears once, with summed score."""
    dense = [_hit("a")]
    sparse = [_hit("a")]
    fused = reciprocal_rank_fusion([(dense, 1.0), (sparse, 1.0)], k=60)
    assert len(fused) == 1
    assert fused[0].score == 2 * (1.0 / (60 + 1))
