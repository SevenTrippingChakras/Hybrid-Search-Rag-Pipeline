"""Q&A endpoint: ask a question, get a grounded answer with citations.

Thin: call the retrieval service (which runs the RAG pipeline) and map the
result into the client-facing shape.
"""

from fastapi import APIRouter

from app.deps import RetrievalServiceDep
from app.models.qa import AskRequest, AskResponse

router = APIRouter(tags=["qa"])


@router.post("/ask")
async def ask(body: AskRequest, service: RetrievalServiceDep) -> AskResponse:
    """Retrieve, then either abstain or answer with citations and confidence."""
    result = await service.answer(body.question)
    return AskResponse.from_pipeline(result)
