"""Tests for the answer confidence scorer (Phase 3.3).

A fake LLM stands in for the completeness judge, so the suite runs offline and
asserts on how the three signals are computed and fused, not model quality.
"""

from hybrid_rag.confidence import ConfidenceScorer
from hybrid_rag.config import settings
from hybrid_rag.models import Answer, Citation, CitationCheck
from hybrid_rag.stores import QueryHit


class _FakeJudge:
    """Returns a scripted completeness decomposition; records the prompt."""

    def __init__(self, parts=((("the question", True),))):
        self._parts = parts
        self.last_user = None

    def parse(self, system, user, schema):
        self.last_user = user
        part_cls = schema.model_fields["parts"].annotation.__args__[0]
        return schema(parts=[part_cls(part=p, addressed=a) for p, a in self._parts])


def _answer(text, cited_numbers=()):
    cites = [Citation(number=n, source="doc.md", text="x") for n in cited_numbers]
    return Answer(query="q", text=text, citations=cites)


def _hit(score):
    return QueryHit("id", "doc", {}, score)


def test_retrieval_confidence_is_mean_of_hit_scores():
    scorer = ConfidenceScorer(llm=_FakeJudge())

    conf = scorer.score(_answer("An answer."), [_hit(1.0), _hit(0.0)], [])

    assert conf.retrieval == 0.5


def test_no_hits_gives_zero_retrieval_confidence():
    conf = ConfidenceScorer(llm=_FakeJudge()).score(_answer("An answer."), [], [])

    assert conf.retrieval == 0.0


def test_citation_coverage_counts_supported_claims_over_all_claims():
    answer = _answer(
        "Reset the pump before restart [1]. Drain the tank first [2]. "
        "An uncited aside.",
        cited_numbers=[1, 2],
    )
    checks = [
        CitationCheck("Reset the pump before restart.", 1, True, "ok"),
        CitationCheck("Drain the tank first.", 2, False, "off-topic"),
    ]

    conf = ConfidenceScorer(llm=_FakeJudge()).score(answer, [_hit(1.0)], checks)

    # 3 claims, only the first is backed by a supported citation.
    assert conf.citation_coverage == 1 / 3


def test_claim_with_any_supported_citation_counts_as_backed():
    answer = _answer("Both steps matter [1][2].", cited_numbers=[1, 2])
    checks = [
        CitationCheck("Both steps matter.", 1, False, "no"),
        CitationCheck("Both steps matter.", 2, True, "yes"),
    ]

    conf = ConfidenceScorer(llm=_FakeJudge()).score(answer, [_hit(1.0)], checks)

    assert conf.citation_coverage == 1.0


def test_completeness_is_fraction_of_addressed_parts():
    judge = _FakeJudge(parts=[("part a", True), ("part b", False)])

    conf = ConfidenceScorer(llm=judge).score(_answer("Ans."), [_hit(1.0)], [])

    assert conf.completeness == 0.5
    assert "QUESTION: q" in judge.last_user


def test_composite_is_the_weighted_sum_of_the_three_signals():
    answer = _answer("Single backed claim [1].", cited_numbers=[1])
    checks = [CitationCheck("Single backed claim.", 1, True, "ok")]
    judge = _FakeJudge(parts=[("only part", True)])

    conf = ConfidenceScorer(llm=judge).score(answer, [_hit(1.0)], checks)

    expected = (
        settings.confidence_retrieval_weight * 1.0
        + settings.confidence_citation_weight * 1.0
        + settings.confidence_completeness_weight * 1.0
    )
    assert conf.retrieval == 1.0
    assert conf.citation_coverage == 1.0
    assert conf.completeness == 1.0
    assert conf.score == expected
