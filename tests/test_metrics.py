"""Tests for the local (no-LLM) eval metrics."""

from hybrid_rag.eval.metrics import (
    abstention_accuracy,
    citation_accuracy,
    retrieval_relevance,
)
from hybrid_rag.models import CitationCheck


def _check(supported):
    return CitationCheck(claim="c", number=1, supported=supported, reason="")


def test_retrieval_relevance_is_gold_recall():
    assert retrieval_relevance(["a.md", "b.md", "x.md"], ["a.md", "b.md"]) == 1.0
    assert retrieval_relevance(["a.md", "x.md"], ["a.md", "b.md"]) == 0.5
    assert retrieval_relevance(["x.md"], ["a.md", "b.md"]) == 0.0


def test_retrieval_relevance_none_without_gold():
    assert retrieval_relevance(["a.md"], []) is None


def test_citation_accuracy_fraction_supported():
    assert citation_accuracy([_check(True), _check(True)]) == 1.0
    assert citation_accuracy([_check(True), _check(False)]) == 0.5
    assert citation_accuracy([_check(False)]) == 0.0


def test_citation_accuracy_none_without_citations():
    assert citation_accuracy([]) is None


def test_abstention_accuracy_only_scores_no_answer():
    assert abstention_accuracy("no_answer", abstained=True) == 1.0
    assert abstention_accuracy("no_answer", abstained=False) == 0.0
    assert abstention_accuracy("multi_hop", abstained=False) is None
    assert abstention_accuracy("multi_hop", abstained=True) is None
