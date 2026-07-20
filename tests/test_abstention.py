"""Tests for the abstention gate (Phase 3.4).

Pure computation over reranked hits -- no LLM, no network. Asserts when the gate
abstains and what the structured NoAnswer carries.
"""

from hybrid_rag.abstention import AbstentionGate
from hybrid_rag.stores import QueryHit


def _hit(score, source="doc.md", heading=None):
    meta = {"source": source}
    if heading is not None:
        meta["heading"] = heading
    return QueryHit("id", "text", meta, score)


def test_answers_when_confidence_at_or_above_threshold():
    gate = AbstentionGate(threshold=0.2)

    assert gate.check("q", [_hit(0.2), _hit(0.2)]) is None


def test_abstains_when_confidence_below_threshold():
    gate = AbstentionGate(threshold=0.5)

    result = gate.check("q", [_hit(0.1), _hit(0.3)])

    assert result is not None
    assert result.retrieval_confidence == 0.2


def test_no_hits_abstains():
    result = AbstentionGate(threshold=0.2).check("q", [])

    assert result is not None
    assert result.retrieval_confidence == 0.0
    assert result.found == []
    assert result.suggested_sources == []


def test_found_labels_use_source_and_heading():
    gate = AbstentionGate(threshold=1.0)

    result = gate.check("q", [_hit(0.1, "a.md", "Intro"), _hit(0.1, "b.md")])

    assert result.found == ["a.md — Intro", "b.md"]


def test_suggested_sources_are_distinct_in_order():
    gate = AbstentionGate(threshold=1.0)

    hits = [_hit(0.1, "a.md"), _hit(0.1, "b.md"), _hit(0.1, "a.md")]
    result = gate.check("q", hits)

    assert result.suggested_sources == ["a.md", "b.md"]


def test_threshold_defaults_to_settings():
    from hybrid_rag.config import settings

    gate = AbstentionGate()
    # A hit just below the configured threshold must abstain.
    result = gate.check("q", [_hit(settings.abstain_threshold - 0.01)])

    assert result is not None
