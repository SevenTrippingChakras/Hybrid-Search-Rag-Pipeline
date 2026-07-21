"""Tests for the grounded generation layer (Phase 3.1).

A fake LLM is injected in place of the provider, so the suite runs offline and
asserts on prompt construction and citation resolution, not model quality.
"""

from app.rag.generation import Generator
from app.rag.stores import QueryHit


class _FakeLLM:
    """Records the prompt it was given and returns a scripted parsed answer."""

    def __init__(self, answer="A grounded answer [1].", citations=(1,)):
        self._answer = answer
        self._citations = list(citations)
        self.last_system = None
        self.last_user = None

    def parse(self, system, user, schema):
        self.last_system = system
        self.last_user = user
        return schema(answer=self._answer, citations=self._citations)


def _hit(id_, doc, source="doc.md", heading=None, page=None):
    meta = {"source": source, "heading": heading, "page": page}
    return QueryHit(id_, doc, meta, 1.0)


def test_generate_resolves_citations_to_sources():
    hits = [
        _hit("a", "Reset the pump before restart.", source="ops.md", heading="Pump"),
        _hit("b", "Unrelated text.", source="misc.md"),
    ]
    llm = _FakeLLM(answer="Reset the pump [1].", citations=[1])

    answer = Generator(llm=llm).generate("how to reset", hits)

    assert answer.text == "Reset the pump [1]."
    assert len(answer.citations) == 1
    cite = answer.citations[0]
    assert cite.number == 1
    assert cite.source == "ops.md"
    assert cite.heading == "Pump"
    assert cite.text == "Reset the pump before restart."


def test_prompt_numbers_context_blocks_with_source_labels():
    hits = [_hit("a", "First passage.", source="a.md", heading="Intro")]
    llm = _FakeLLM()

    Generator(llm=llm).generate("q", hits)

    assert "[1] (a.md — Intro)" in llm.last_user
    assert "First passage." in llm.last_user
    assert "Question: q" in llm.last_user


def test_out_of_range_citations_are_dropped():
    hits = [_hit("a", "Only passage.")]
    llm = _FakeLLM(answer="Text [1][5].", citations=[1, 5])

    answer = Generator(llm=llm).generate("q", hits)

    assert [c.number for c in answer.citations] == [1]


def test_empty_citations_yields_uncited_answer():
    hits = [_hit("a", "Some context.")]
    llm = _FakeLLM(answer="The context does not cover this.", citations=[])

    answer = Generator(llm=llm).generate("q", hits)

    assert answer.citations == []
    assert "does not cover" in answer.text
