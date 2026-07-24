"""Contextual Retrieval (the Anthropic technique).

An isolated chunk is often ambiguous -- "the 5-day window" says nothing about
which policy it belongs to. Before embedding, prepend a one-to-two sentence,
LLM-generated blurb that situates the chunk in its whole document, so the dense
embedding and the BM25 index both see the disambiguated text and stop confusing
similar-looking chunks from different sections.

This is an enhancement, not a chunking strategy: it runs on the ingest write path
*after* any strategy has split the document, right before indexing. One shared
step covers ``fixed``/``header``/``semantic`` alike.
"""

from pydantic import BaseModel

from app.rag.llm import LLM, build_llm
from app.rag.models import Chunk

SYSTEM_PROMPT = (
    "You situate a text chunk within its source document to improve search "
    "retrieval. Given the whole document and one chunk taken from it, write a "
    "short one-to-two sentence context stating which part of the document the "
    "chunk is from and what it covers. Answer with only that context, nothing "
    "else."
)


class _Context(BaseModel):
    """The situating blurb the model returns for one chunk."""

    context: str


def contextualize(
    chunks: list[Chunk], document: str, llm: LLM | None = None
) -> list[Chunk]:
    """Prepend a document-situating blurb to each chunk's text, in place.

    ``document`` is the full source text the chunks came from. Each chunk's
    retrieval text becomes ``blurb + original``; ``char_count`` is refreshed to
    match. Returns the same list for convenience.
    """
    llm = llm or build_llm()
    for chunk in chunks:
        blurb = _situate(llm, document, chunk.text)
        chunk.text = f"{blurb}\n{chunk.text}"
        chunk.char_count = len(chunk.text)
    return chunks


def _situate(llm: LLM, document: str, chunk_text: str) -> str:
    """Ask the LLM for a short blurb placing this chunk within the document."""
    user = (
        f"<document>\n{document}\n</document>\n\n"
        "Here is the chunk to situate within the document:\n"
        f"<chunk>\n{chunk_text}\n</chunk>"
    )
    return llm.parse(SYSTEM_PROMPT, user, _Context).context.strip()
