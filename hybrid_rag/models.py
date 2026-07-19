"""Shared data models for the RAG pipeline."""

from dataclasses import dataclass


@dataclass
class Segment:
    """A normalized region of a source document with structural metadata.

    The loader's output unit: clean plaintext plus whatever structure the source
    exposes (a heading, a PDF page number). Chunking operates on segments.
    """

    text: str
    source: str
    heading: str | None = None
    page: int | None = None


@dataclass
class Chunk:
    """A retrieval unit produced by a chunking strategy.

    Carries citation provenance (source, heading, page) and index bookkeeping
    (chunk_index, strategy, char_count). ``strategy`` records which chunker made it.
    """

    text: str
    source: str
    chunk_index: int
    strategy: str
    char_count: int
    heading: str | None = None
    page: int | None = None
