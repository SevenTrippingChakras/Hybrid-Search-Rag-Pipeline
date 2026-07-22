"""Q&A request/response schemas for ``POST /ask``.

The client-facing contract for asking a question. ``AskResponse.from_pipeline``
maps the RAG pipeline's rich result into this stable shape - either a grounded
answer with citations/confidence, or a graceful abstention.
"""

from pydantic import BaseModel

from app.rag.pipeline import PipelineResult


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    number: int
    source: str
    text: str
    heading: str | None = None
    page: int | None = None


class ConfidenceOut(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    score: float


class SourceOut(BaseModel):
    source: str | None = None
    document_id: str | None = None
    score: float


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
        sources = [
            SourceOut(
                source=h.metadata.get("source"),
                document_id=h.metadata.get("document_id"),
                score=h.score,
            )
            for h in result.hits
        ]
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
                CitationOut(
                    number=c.number,
                    source=c.source,
                    text=c.text,
                    heading=c.heading,
                    page=c.page,
                )
                for c in (answer.citations if answer else [])
            ],
            confidence=(
                ConfidenceOut(
                    retrieval=confidence.retrieval,
                    citation_coverage=confidence.citation_coverage,
                    completeness=confidence.completeness,
                    score=confidence.score,
                )
                if confidence
                else None
            ),
            abstained=False,
            sources=sources,
        )
