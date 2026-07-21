"""RAGAS judge wiring.

RAGAS needs an LLM (the judge) and an embeddings model. Both are built from the
same OpenAI config the app uses, so eval and app share one provider. The metrics
score via async ``ascore``, so the client is ``AsyncOpenAI``.
"""

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory

from hybrid_rag.config import settings
from hybrid_rag.embeddings import EMBED_MODEL


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


def build_judge_llm():
    """The OpenAI-backed LLM the RAGAS metrics use to grade answers."""
    return llm_factory(settings.generation_model, provider="openai", client=_client())


def build_judge_embeddings() -> OpenAIEmbeddings:
    """OpenAI embeddings for metrics with a semantic-similarity component."""
    return OpenAIEmbeddings(client=_client(), model=EMBED_MODEL)
