"""Generation: the grounded answer layer (Phase 3.1).

``Generator`` turns retrieved chunks into an answer that cites its sources with
bracketed ``[n]`` markers and refuses to go beyond the given context. It depends
on the ``LLM`` port, so the provider is a config choice, not an architecture one.
"""

from pydantic import BaseModel

from hybrid_rag.llm import LLM, build_llm
from hybrid_rag.models import Answer, Citation
from hybrid_rag.stores import QueryHit

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
    """The LLM boundary schema: prose with inline ``[n]`` plus the numbers used."""

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
