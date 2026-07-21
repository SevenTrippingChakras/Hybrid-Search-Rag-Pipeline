"""Offline tests for the Pipeline orchestrator: wiring and the abstention branch.

All collaborators are fakes -- no network, no OpenSearch, no LLM. These assert
the pipeline calls its stages in order and routes correctly, not what any single
component computes (those have their own tests).
"""

from hybrid_rag.models import Answer, CitationCheck, Confidence, NoAnswer
from hybrid_rag.pipeline import Pipeline
from hybrid_rag.stores import QueryHit


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.searched = None

    def search(self, question):
        self.searched = question
        return self._hits


class _FakeGate:
    def __init__(self, no_answer):
        self._no_answer = no_answer

    def check(self, query, hits):
        return self._no_answer


class _FakeGenerator:
    def __init__(self, answer):
        self._answer = answer
        self.called = False

    def generate(self, query, hits):
        self.called = True
        return self._answer


class _FakeVerifier:
    def __init__(self, checks):
        self._checks = checks

    def verify(self, answer):
        return self._checks


class _FakeScorer:
    def __init__(self, confidence):
        self._confidence = confidence

    def score(self, answer, hits, checks):
        return self._confidence


_HITS = [QueryHit("c1", "Paris is the capital of France.", {"source": "d.md"}, 0.9)]


def test_generate_path_wires_all_stages():
    answer = Answer(query="q", text="Paris.[1]", citations=[])
    checks = [CitationCheck(claim="Paris.", number=1, supported=True, reason="ok")]
    confidence = Confidence(
        retrieval=0.9, citation_coverage=1.0, completeness=1.0, score=0.94
    )
    generator = _FakeGenerator(answer)

    pipeline = Pipeline(
        retriever=_FakeRetriever(_HITS),
        gate=_FakeGate(None),
        generator=generator,
        verifier=_FakeVerifier(checks),
        scorer=_FakeScorer(confidence),
    )
    result = pipeline.answer("What is the capital of France?")

    assert not result.abstained
    assert result.answer is answer
    assert result.checks == checks
    assert result.confidence is confidence
    assert result.hits == _HITS
    assert generator.called


def test_abstention_path_skips_generation():
    no_answer = NoAnswer(
        query="q",
        retrieval_confidence=0.05,
        message="not enough",
        found=[],
        suggested_sources=[],
    )
    generator = _FakeGenerator(Answer(query="q", text="should not run", citations=[]))

    pipeline = Pipeline(
        retriever=_FakeRetriever(_HITS),
        gate=_FakeGate(no_answer),
        generator=generator,
        verifier=_FakeVerifier([]),
        scorer=_FakeScorer(None),
    )
    result = pipeline.answer("Unanswerable question?")

    assert result.abstained
    assert result.no_answer is no_answer
    assert result.answer is None
    assert not generator.called
