"""Tests for citation verification (Phase 3.2).

A fake judge is injected in place of the LLM, so the suite runs offline and
asserts on claim/citation pairing and verdict wiring, not model quality.
"""

import time

from app.rag.models import Answer, Citation
from app.rag.verification import CitationVerifier


class _FakeJudge:
    """Returns a scripted verdict; records each (claim, passage) it was asked."""

    def __init__(self, supported=True, reason="ok"):
        self._supported = supported
        self._reason = reason
        self.prompts = []

    def parse(self, system, user, schema):
        self.prompts.append(user)
        return schema(supported=self._supported, reason=self._reason)


def _answer(text, citations):
    cites = [Citation(number=n, source="doc.md", text=t) for n, t in citations]
    return Answer(query="q", text=text, citations=cites)


def test_verify_pairs_each_claim_with_its_citation():
    answer = _answer(
        "Reset the pump before restart [1]. Drain the tank first [2].",
        [(1, "Reset the pump before restart."), (2, "Drain the tank first.")],
    )
    judge = _FakeJudge(supported=True)

    checks = CitationVerifier(llm=judge).verify(answer)

    assert [c.number for c in checks] == [1, 2]
    assert checks[0].claim == "Reset the pump before restart."
    assert all(c.supported for c in checks)
    # Each pair sends its own claim and passage to the judge. Pairs run
    # concurrently, so assert membership, not position (result order is
    # checked above via [c.number for c in checks] == [1, 2]).
    joined = "\n".join(judge.prompts)
    assert "Reset the pump before restart." in joined
    assert "Drain the tank first." in joined


def test_unsupported_citation_is_flagged():
    answer = _answer("The sky is green [1].", [(1, "Grass is green.")])
    judge = _FakeJudge(supported=False, reason="passage is about grass, not sky")

    checks = CitationVerifier(llm=judge).verify(answer)

    assert len(checks) == 1
    assert checks[0].supported is False
    assert "grass" in checks[0].reason


def test_sentence_with_two_markers_yields_two_checks():
    answer = _answer(
        "Both steps matter [1][2].",
        [(1, "Step one."), (2, "Step two.")],
    )
    judge = _FakeJudge()

    checks = CitationVerifier(llm=judge).verify(answer)

    assert [c.number for c in checks] == [1, 2]
    assert all(c.claim == "Both steps matter." for c in checks)


def test_marker_without_resolved_citation_is_skipped():
    answer = _answer("Claim one [1]. Claim two [5].", [(1, "Backs claim one.")])
    judge = _FakeJudge()

    checks = CitationVerifier(llm=judge).verify(answer)

    assert [c.number for c in checks] == [1]


def test_uncited_answer_yields_no_checks():
    answer = _answer("The context does not cover this.", [])
    judge = _FakeJudge()

    checks = CitationVerifier(llm=judge).verify(answer)

    assert checks == []


class _SlowJudge:
    """Sleeps on every parse, to prove the calls overlap rather than serialize."""

    def __init__(self, delay=0.05):
        self._delay = delay

    def parse(self, system, user, schema):
        time.sleep(self._delay)
        return schema(supported=True, reason="ok")


def test_pairs_are_judged_concurrently():
    answer = _answer(
        "A [1]. B [2]. C [3]. D [4]. E [5].",
        [(n, f"passage {n}") for n in range(1, 6)],
    )

    start = time.perf_counter()
    checks = CitationVerifier(llm=_SlowJudge(delay=0.05)).verify(answer)
    elapsed = time.perf_counter() - start

    assert [c.number for c in checks] == [1, 2, 3, 4, 5]
    # Sequential would be 5 x 50ms = 250ms; concurrent is ~one delay plus overhead.
    assert elapsed < 0.15
