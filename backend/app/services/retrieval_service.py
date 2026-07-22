"""Q&A read side: run a question through the RAG pipeline.

Thin wrapper over ``Pipeline.answer``. The pipeline is synchronous and heavy
(embeddings, cross-encoder rerank, LLM generation + verification), so it runs in
a thread to keep the event loop free. The pipeline itself is built once and
injected (its reranker model is expensive to load).
"""

import asyncio

from app.rag.pipeline import Pipeline, PipelineResult


class RetrievalService:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    async def answer(self, question: str) -> PipelineResult:
        return await asyncio.to_thread(self._pipeline.answer, question)
