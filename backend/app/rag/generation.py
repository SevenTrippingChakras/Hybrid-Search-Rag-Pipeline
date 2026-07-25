"""Grounded answer generation.

``Generator`` turns retrieved chunks into an answer that cites its sources with
bracketed ``[n]`` markers and refuses to go beyond the given context. It depends
on the ``LLM`` port, so the provider is a config choice.
"""

import re
from collections.abc import Iterator

from pydantic import BaseModel

from app.rag.llm import LLM, build_llm
from app.rag.models import Answer, Citation
from app.rag.stores import QueryHit

_MARKER = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Follow these rules exactly:\n"
    "- Use only facts stated in the context. Never rely on outside knowledge.\n"
    "- Cite every claim with the bracketed number of the passage that supports "
    "it, e.g. [1] or [2][3]. Place the marker right after the claim.\n"
    "- List, in the citations field, every passage number you cited.\n"
    "- If the context does not contain enough information to answer, say so "
    "plainly and leave the citations field empty. Do not guess."
)


class _LLMAnswer(BaseModel):
    """The LLM's reply: prose with inline ``[n]`` plus the numbers used."""

    answer: str
    citations: list[int]


class Generator:
    """Builds the grounded prompt, calls the LLM, resolves cites to sources."""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm or build_llm()

    def generate(self, query: str, hits: list[QueryHit]) -> Answer:
        """Answer ``query`` from ``hits``, grounded and cited."""
        user = self._build_prompt(query, hits)
        result = self._llm.parse(SYSTEM_PROMPT, user, _LLMAnswer)
        return Answer(
            query=query,
            text=result.answer,
            citations=self._resolve_citations(result.citations, hits),
        )

    def stream(self, query: str, hits: list[QueryHit]) -> Iterator[str]:
        """Yield the grounded answer as text deltas, markers included."""
        yield from self._llm.stream(SYSTEM_PROMPT, self._build_prompt(query, hits))

    def build_answer(self, query: str, text: str, hits: list[QueryHit]) -> Answer:
        """Assemble the final ``Answer`` from streamed prose.

        The cited passage numbers are read back from the ``[n]`` markers in the
        text, since the streaming path has no separate structured citations field.
        """
        numbers = sorted({int(n) for n in _MARKER.findall(text)})
        return Answer(
            query=query,
            text=text,
            citations=self._resolve_citations(numbers, hits),
        )

    @staticmethod
    def _build_prompt(query: str, hits: list[QueryHit]) -> str:
        """Render the question and the hits as numbered context blocks."""
        blocks = []
        for number, hit in enumerate(hits, start=1):
            source = hit.metadata.get("source", "unknown")
            heading = hit.metadata.get("heading")
            label = f"{source} — {heading}" if heading else source
            blocks.append(f"[{number}] ({label})\n{hit.document}")
        context = "\n\n".join(blocks)
        return f"Question: {query}\n\nContext:\n{context}"

    @staticmethod
    def _resolve_citations(numbers: list[int], hits: list[QueryHit]) -> list[Citation]:
        """Map each cited passage number back to the chunk that backs it."""
        citations = []
        for number in numbers:
            if 1 <= number <= len(hits):
                hit = hits[number - 1]
                citations.append(
                    Citation(
                        number=number,
                        source=hit.metadata.get("source", "unknown"),
                        text=hit.document,
                        heading=hit.metadata.get("heading"),
                        page=hit.metadata.get("page"),
                    )
                )
        return citations
