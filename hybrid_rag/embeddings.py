"""OpenAI embedding helper.

A thin wrapper around the embeddings endpoint. The client is created lazily on
first use so importing this module never requires an API key.
"""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        _client = OpenAI()
    return _client


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input."""
    response = _get_client().embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
