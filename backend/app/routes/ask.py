"""Q&A endpoint: ask a question, get a grounded answer with citations.

Two shapes over the same pipeline: ``POST /ask`` returns the whole answer at
once (for eval and programmatic callers), while ``POST /ask/stream`` streams the
answer over Server-Sent Events so the UI can render tokens as they arrive.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps import RetrievalServiceDep
from app.models.qa import (
    AbstainEvent,
    AskRequest,
    AskResponse,
    DeltaEvent,
    FinalEvent,
    SourcesEvent,
    SSEEvent,
)
from app.rag.pipeline import StreamAbstain, StreamDelta, StreamFinal, StreamStart
from app.services.retrieval_service import RetrievalService

router = APIRouter(tags=["qa"])

# Proxies (nginx and friends) buffer responses by default, which defeats
# streaming; these headers tell them and the browser to flush immediately.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/ask")
async def ask(body: AskRequest, service: RetrievalServiceDep) -> AskResponse:
    """Retrieve, then either abstain or answer with citations and confidence."""
    result = await service.answer(body.question)
    return AskResponse.from_pipeline(result)


@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest, service: RetrievalServiceDep
) -> StreamingResponse:
    """Stream the answer as SSE: sources, then prose deltas, then a final event."""
    return StreamingResponse(
        _sse_events(body.question, service),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _sse_events(question: str, service: RetrievalService) -> AsyncIterator[str]:
    """Map pipeline stream events onto the SSE wire format, one frame each."""
    async for event in service.answer_stream(question):
        if isinstance(event, StreamStart):
            yield _frame(SourcesEvent.of(event.hits))
        elif isinstance(event, StreamDelta):
            yield _frame(DeltaEvent(text=event.text))
        elif isinstance(event, StreamAbstain):
            message = event.no_answer.message if event.no_answer else None
            yield _frame(
                AbstainEvent(
                    message=message,
                    sources=SourcesEvent.of(event.hits).sources,
                )
            )
        elif isinstance(event, StreamFinal):
            yield _frame(FinalEvent.of(event.answer, event.confidence))


def _frame(payload: SSEEvent) -> str:
    """Render one streamed event as a named SSE frame."""
    return f"event: {payload.EVENT}\ndata: {payload.model_dump_json()}\n\n"
