"""The LLM behind one interface.

``LLM`` is the port, ``OpenAILLM`` the adapter, selected by ``build_llm`` from
the ``LLM_BACKEND`` env var. Callers (generation, later citation verification)
depend on the port, so adding a provider (Anthropic, Bedrock) is a new adapter
plus a config value, never a change to the RAG logic. Structured-output
mechanics are absorbed here so provider differences never leak upward.
"""

from collections.abc import Iterator
from typing import Protocol, TypeVar, runtime_checkable

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLM(Protocol):
    """The LLM port: structured parse plus token streaming of free-form text."""

    def parse(self, system: str, user: str, schema: type[T]) -> T: ...

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield answer text in deltas as the model produces them."""
        ...


class OpenAILLM:
    """OpenAI chat model with Structured Outputs, client created lazily."""

    def __init__(self, model: str = settings.generation_model) -> None:
        self._model = model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def parse(self, system: str, user: str, schema: type[T]) -> T:
        """Constrain the model to ``schema`` and return the parsed object."""
        completion = self._get_client().chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            raise ValueError(f"model refused to answer: {message.refusal}")
        return message.parsed

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream the model's reply as free-form text deltas.

        Structured Outputs and token streaming don't compose, so the streaming
        path takes plain text; callers recover citations from the ``[n]``
        markers the prompt instructs the model to emit inline.
        """
        stream = self._get_client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_llm() -> LLM:
    """Return the configured LLM adapter (only ``openai`` for now)."""
    if settings.llm_backend == "openai":
        return OpenAILLM()
    raise ValueError(f"unknown llm_backend: {settings.llm_backend}")
