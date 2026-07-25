"""Q&A request/response schemas for ``POST /ask``.

The client-facing contract for asking a question. ``AskResponse.from_pipeline``
maps the RAG pipeline's rich result into this stable shape - either a grounded
answer with citations/confidence, or a graceful abstention.
"""

from typing import ClassVar

from pydantic import BaseModel

from app.rag.models import Answer, Citation, Confidence
from app.rag.pipeline import PipelineResult
from app.rag.stores import QueryHit


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    number: int
    source: str
    text: str
    heading: str | None = None
    page: int | None = None

    @classmethod
    def from_citation(cls, c: Citation) -> "CitationOut":
        return cls(
            number=c.number,
            source=c.source,
            text=c.text,
            heading=c.heading,
            page=c.page,
        )


class ConfidenceOut(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    score: float

    @classmethod
    def from_confidence(cls, c: Confidence) -> "ConfidenceOut":
        return cls(
            retrieval=c.retrieval,
            citation_coverage=c.citation_coverage,
            completeness=c.completeness,
            score=c.score,
        )


class SourceOut(BaseModel):
    source: str | None = None
    document_id: str | None = None
    score: float

    @classmethod
    def from_hit(cls, h: QueryHit) -> "SourceOut":
        return cls(
            source=h.metadata.get("source"),
            document_id=h.metadata.get("document_id"),
            score=h.score,
        )


class AskResponse(BaseModel):
    query: str
    answer: str | None
    citations: list[CitationOut]
    confidence: ConfidenceOut | None
    abstained: bool
    message: str | None = None
    sources: list[SourceOut]

    @classmethod
    def from_pipeline(cls, result: PipelineResult) -> "AskResponse":
        sources = [SourceOut.from_hit(h) for h in result.hits]
        if result.abstained:
            na = result.no_answer
            return cls(
                query=result.query,
                answer=None,
                citations=[],
                confidence=None,
                abstained=True,
                message=na.message if na else None,
                sources=sources,
            )

        answer = result.answer
        confidence = result.confidence
        return cls(
            query=result.query,
            answer=answer.text if answer else None,
            citations=[
                CitationOut.from_citation(c)
                for c in (answer.citations if answer else [])
            ],
            confidence=(
                ConfidenceOut.from_confidence(confidence) if confidence else None
            ),
            abstained=False,
            sources=sources,
        )


# --- Streaming (SSE) payloads ----------------------------------------------
# One model per SSE event type. The event name on the wire is the subclass's
# ``EVENT`` and the JSON body is the model itself.


class SSEEvent(BaseModel):
    """Base for streamed events: subclasses set ``EVENT`` to their wire name."""

    EVENT: ClassVar[str]


class SourcesEvent(SSEEvent):
    """Sent once when answering begins: the retrieved sources."""

    EVENT: ClassVar[str] = "sources"
    sources: list[SourceOut]

    @classmethod
    def of(cls, hits: list[QueryHit]) -> "SourcesEvent":
        return cls(sources=[SourceOut.from_hit(h) for h in hits])


class DeltaEvent(SSEEvent):
    """A chunk of answer prose."""

    EVENT: ClassVar[str] = "delta"
    text: str


class FinalEvent(SSEEvent):
    """Sent once at the end: full answer text, resolved citations, confidence."""

    EVENT: ClassVar[str] = "final"
    answer: str
    citations: list[CitationOut]
    confidence: ConfidenceOut | None

    @classmethod
    def of(cls, answer: Answer, confidence: Confidence | None) -> "FinalEvent":
        return cls(
            answer=answer.text,
            citations=[CitationOut.from_citation(c) for c in answer.citations],
            confidence=(
                ConfidenceOut.from_confidence(confidence) if confidence else None
            ),
        )


class AbstainEvent(SSEEvent):
    """Sent instead of any answer when retrieval falls short."""

    EVENT: ClassVar[str] = "abstain"
    message: str | None
    sources: list[SourceOut]


class ErrorEvent(SSEEvent):
    """Sent if the pipeline raises mid-stream."""

    EVENT: ClassVar[str] = "error"
    message: str
