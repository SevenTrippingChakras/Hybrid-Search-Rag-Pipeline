"""Q&A read side: run a question through the RAG pipeline.

Thin wrapper over ``Pipeline``. The pipeline is synchronous and heavy
(embeddings, cross-encoder rerank, LLM generation + verification), so it runs in
a thread to keep the event loop free. The pipeline itself is built once and
injected (its reranker model is expensive to load).
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator

from app.rag.pipeline import Pipeline, PipelineResult, StreamEvent

_DONE = object()  # sentinel marking the end of the bridged stream


class RetrievalService:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    async def answer(self, question: str) -> PipelineResult:
        return await asyncio.to_thread(self._pipeline.answer, question)

    async def answer_stream(self, question: str) -> AsyncIterator[StreamEvent]:
        """Stream pipeline events, draining the blocking generator off-loop."""
        gen = lambda: self._pipeline.answer_stream(question)  # noqa: E731
        async for event in _bridge(gen):
            yield event


async def _bridge[T](make_gen: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Run a blocking generator in a worker thread, yield its items on the loop.

    The producer pushes each item onto a queue via ``call_soon_threadsafe`` so
    the event loop stays free while the CPU/IO-heavy pipeline runs. Exceptions
    raised inside the generator are forwarded and re-raised to the consumer.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        try:
            for item in make_gen():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # surface pipeline failures to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    task = loop.run_in_executor(None, produce)
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        await task
