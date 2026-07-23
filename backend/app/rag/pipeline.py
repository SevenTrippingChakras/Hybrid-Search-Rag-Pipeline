"""The read pipeline, composed end to end.

``Pipeline`` wires the standalone components into one ``answer(question)`` call:
retrieve, abstain, generate, verify, score. Every collaborator is injectable so
the wiring can be tested with fakes; the defaults build the real components.
"""

from dataclasses import dataclass, field

from app.rag.abstention import AbstentionGate
from app.rag.confidence import ConfidenceScorer
from app.rag.generation import Generator
from app.rag.models import Answer, CitationCheck, Confidence, NoAnswer
from app.rag.retrieval import Retriever
from app.rag.stores import QueryHit
from app.rag.verification import CitationVerifier


@dataclass
class PipelineResult:
    """Everything one question produced: the answer or a graceful abstention."""

    query: str
    hits: list[QueryHit]
    answer: Answer | None = None
    checks: list[CitationCheck] = field(default_factory=list)
    confidence: Confidence | None = None
    no_answer: NoAnswer | None = None

    @property
    def abstained(self) -> bool:
        return self.no_answer is not None


class Pipeline:
    """Composes retrieval, abstention, generation, verification, and scoring."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        gate: AbstentionGate | None = None,
        generator: Generator | None = None,
        verifier: CitationVerifier | None = None,
        scorer: ConfidenceScorer | None = None,
        index: str | None = None,
    ) -> None:
        self._retriever = retriever or Retriever(index=index)
        self._gate = gate or AbstentionGate()
        self._generator = generator or Generator()
        self._verifier = verifier or CitationVerifier()
        self._scorer = scorer or ConfidenceScorer()

    def warmup(self) -> None:
        """Pre-load heavy models (the cross-encoder reranker) before serving."""
        self._retriever.warmup()

    def answer(self, question: str) -> PipelineResult:
        """Retrieve, then either abstain or generate a verified, scored answer."""
        hits = self._retriever.search(question)

        no_answer = self._gate.check(question, hits)
        if no_answer is not None:
            return PipelineResult(query=question, hits=hits, no_answer=no_answer)

        answer = self._generator.generate(question, hits)
        checks = self._verifier.verify(answer)
        confidence = self._scorer.score(answer, hits, checks)
        return PipelineResult(
            query=question,
            hits=hits,
            answer=answer,
            checks=checks,
            confidence=confidence,
        )
