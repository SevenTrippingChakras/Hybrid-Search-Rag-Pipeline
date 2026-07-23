"""Answer confidence scoring.

An answer is only as trustworthy as its grounding. ``ConfidenceScorer`` folds
three independent signals into one composite score:

- retrieval confidence: how relevant were the chunks fed to the generator,
- citation coverage: what fraction of the answer's claims carry a verified
  (supported) citation,
- completeness: how much of the question the answer actually addressed.

Retrieval and coverage are read off earlier stages (rerank scores, the Phase 3.2
citation checks); only completeness needs reasoning, so this depends on the
``LLM`` port, symmetric with generation and verification.
"""

from pydantic import BaseModel

from app.config import settings
from app.rag.llm import LLM, build_llm
from app.rag.models import Answer, CitationCheck, Confidence
from app.rag.stores import QueryHit
from app.rag.verification import split_claims


def retrieval_confidence(hits: list[QueryHit]) -> float:
    """Relevance of the single best reranked chunk (cross-encoder score, 0-1).

    The top-1, not the mean: cross-encoder rerank scores are spiky -- one strongly
    relevant chunk scores near 1 while the rest sit near 0 -- and ``top_k`` pads
    the list with low-relevance chunks, so a mean is dominated by how many padding
    chunks there are (a config value) rather than by grounding. One strong chunk is
    enough to answer, so both the abstention gate and the composite confidence read
    the best hit.
    """
    if not hits:
        return 0.0
    return max(hit.score for hit in hits)


COMPLETENESS_SYSTEM = (
    "You judge how completely an ANSWER addresses a QUESTION. "
    "Break the question into its distinct parts (each sub-question, requested "
    "item, or aspect). For each part, decide whether the answer addresses it.\n"
    "- A part is addressed only if the answer gives a substantive response to it.\n"
    "- Judge coverage of the question, not correctness or sourcing."
)


class _Part(BaseModel):
    """One distinct part of the question and whether the answer addresses it."""

    part: str
    addressed: bool


class _Completeness(BaseModel):
    """The judge's decomposition of the question into addressed/unaddressed parts."""

    parts: list[_Part]


class ConfidenceScorer:
    """Scores an answer across retrieval, citation coverage, and completeness."""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm or build_llm()

    def score(
        self,
        answer: Answer,
        hits: list[QueryHit],
        checks: list[CitationCheck],
    ) -> Confidence:
        """Fold the three trust signals into a weighted composite confidence."""
        retrieval = retrieval_confidence(hits)
        coverage = self._citation_coverage(answer, checks)
        completeness = self._completeness(answer)
        composite = (
            settings.confidence_retrieval_weight * retrieval
            + settings.confidence_citation_weight * coverage
            + settings.confidence_completeness_weight * completeness
        )
        return Confidence(
            retrieval=retrieval,
            citation_coverage=coverage,
            completeness=completeness,
            score=composite,
        )

    @staticmethod
    def _citation_coverage(answer: Answer, checks: list[CitationCheck]) -> float:
        """Fraction of the answer's claims backed by a supported citation."""
        claims = split_claims(answer.text)
        if not claims:
            return 0.0
        supported = {check.claim for check in checks if check.supported}
        backed = sum(1 for claim in claims if claim in supported)
        return backed / len(claims)

    def _completeness(self, answer: Answer) -> float:
        """Ask the judge which parts of the question the answer addresses."""
        verdict = self._llm.parse(
            COMPLETENESS_SYSTEM,
            f"QUESTION: {answer.query}\n\nANSWER:\n{answer.text}",
            _Completeness,
        )
        if not verdict.parts:
            return 0.0
        addressed = sum(1 for part in verdict.parts if part.addressed)
        return addressed / len(verdict.parts)
