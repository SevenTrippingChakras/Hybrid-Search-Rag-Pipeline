"""Tests for Contextual Retrieval (the Anthropic technique).

A fake LLM stands in for the provider, so the suite runs offline and asserts on
what gets prepended and what the model is shown -- not model quality.
"""

from app.rag.contextual import contextualize
from app.rag.models import Chunk


class _FakeLLM:
    """Returns a scripted blurb and records the prompts it was given."""

    def __init__(self, blurb="From the Refunds section of the billing policy."):
        self._blurb = blurb
        self.users: list[str] = []

    def parse(self, system, user, schema):
        self.users.append(user)
        return schema(context=self._blurb)


def _chunk(text, index=0):
    return Chunk(
        text=text,
        source="policy.md",
        chunk_index=index,
        strategy="header",
        char_count=len(text),
    )


def test_contextualize_prepends_blurb_and_refreshes_char_count():
    chunk = _chunk("Refunds are issued within 5 days.")
    llm = _FakeLLM("From the Refunds section of the billing policy.")

    out = contextualize([chunk], document="full policy text", llm=llm)

    assert out[0].text == (
        "From the Refunds section of the billing policy.\n"
        "Refunds are issued within 5 days."
    )
    assert out[0].char_count == len(out[0].text)


def test_contextualize_shows_llm_the_document_and_the_chunk():
    llm = _FakeLLM()

    contextualize([_chunk("Refunds within 5 days.")], document="WHOLE DOC", llm=llm)

    prompt = llm.users[0]
    assert "WHOLE DOC" in prompt
    assert "Refunds within 5 days." in prompt


def test_contextualize_calls_the_llm_once_per_chunk():
    llm = _FakeLLM()

    contextualize(
        [_chunk("a", 0), _chunk("b", 1), _chunk("c", 2)], document="doc", llm=llm
    )

    assert len(llm.users) == 3


def test_contextualize_empty_list_is_a_noop():
    llm = _FakeLLM()
    assert contextualize([], document="doc", llm=llm) == []
    assert llm.users == []
