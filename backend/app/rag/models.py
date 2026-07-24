"""Shared data models for the RAG pipeline."""

from dataclasses import dataclass


@dataclass
class Segment:
    """A normalized region of a source document: clean text plus metadata."""

    text: str
    source: str
    heading: str | None = None
    page: int | None = None


@dataclass
class Chunk:
    """A retrieval unit from a chunking strategy, with provenance and bookkeeping."""

    text: str
    source: str
    chunk_index: int
    strategy: str
    char_count: int
    heading: str | None = None
    page: int | None = None
    document_id: str | None = None
    element_type: str | None = None


@dataclass
class Citation:
    """A ``[number]`` marker resolved back to the chunk that backs it."""

    number: int
    source: str
    text: str
    heading: str | None = None
    page: int | None = None


@dataclass
class Answer:
    """A grounded answer: prose with inline ``[n]`` markers plus resolved cites."""

    query: str
    text: str
    citations: list[Citation]


@dataclass
class CitationCheck:
    """A judge's verdict on whether a cited passage supports a claim."""

    claim: str
    number: int
    supported: bool
    reason: str


@dataclass
class NoAnswer:
    """A graceful abstention: what was found, why it fell short, where to look.

    Returned instead of an ``Answer`` when retrieval confidence is below the
    threshold, so the system says what it could not answer rather than guessing.
    """

    query: str
    retrieval_confidence: float
    message: str
    found: list[str]
    suggested_sources: list[str]


@dataclass
class Confidence:
    """How trustworthy an answer is, per signal plus a weighted composite."""

    retrieval: float
    citation_coverage: float
    completeness: float
    score: float
