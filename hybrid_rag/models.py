"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """A normalized region of a source document with structural metadata.

    A segment is the loader's output unit: clean plaintext plus whatever
    structure the source format exposes (a markdown/HTML heading, a PDF page
    number). Chunking (Phase 1.2) operates on segments, not raw files.
    """

    text: str
    source: str
    heading: str | None = None
    page: int | None = None


@dataclass
class Chunk:
    """A retrieval unit produced by a chunking strategy.

    Carries the provenance a citation needs (source, heading, page) plus the
    bookkeeping the index needs (chunk_index, strategy, char_count). ``strategy``
    records which chunker made it so strategies can be compared (Phase 4.3).
    """

    text: str
    source: str
    chunk_index: int
    strategy: str
    char_count: int
    heading: str | None = None
    page: int | None = None
