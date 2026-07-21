"""Configurable chunking strategies.

Three switchable strategies turn a document's loader `Segment`s into `Chunk`s:

- ``fixed``    — sliding character window with overlap (structure-blind baseline)
- ``header``   — recursive splitting that respects heading sections (structure-aware)
- ``semantic`` — splits on topic boundaries via embedding similarity

Every chunk records which strategy produced it, so strategies can be compared.
Each function expects the segments of a single document.
"""

import re
from collections.abc import Callable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embeddings import embed_texts
from app.rag.models import Chunk, Segment

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

EmbedFn = Callable[[list[str]], list[list[float]]]


def chunk(segments: list[Segment], strategy: str = "fixed", **params) -> list[Chunk]:
    """Dispatch to a chunking strategy by name."""
    if strategy == "fixed":
        return chunk_fixed(segments, **params)
    if strategy == "header":
        return chunk_by_header(segments, **params)
    if strategy == "semantic":
        return chunk_semantic(segments, **params)
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_fixed(
    segments: list[Segment], *, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Baseline: fixed-size character windows over the whole document.

    Ignores document structure on purpose so it can serve as the control when
    comparing against the structure-aware strategies.
    """
    source = segments[0].source if segments else ""
    text = "\n\n".join(seg.text for seg in segments)
    pieces = _sliding_window(text, chunk_size, overlap)
    return [_chunk(p, source, i, "fixed") for i, p in enumerate(pieces)]


def chunk_by_header(
    segments: list[Segment], *, chunk_size: int = 800, overlap: int = 150
) -> list[Chunk]:
    """Structure-aware: split within each heading section, never across.

    Each segment already corresponds to one heading section (from the loader).
    A section that exceeds ``chunk_size`` is split recursively on natural
    boundaries (paragraph, line, word), preserving the section's heading/page.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks: list[Chunk] = []
    for seg in segments:
        for piece in splitter.split_text(seg.text):
            chunks.append(
                _chunk(piece, seg.source, len(chunks), "header", seg.heading, seg.page)
            )
    return chunks


def chunk_semantic(
    segments: list[Segment],
    *,
    embed_fn: EmbedFn | None = None,
    breakpoint_percentile: int = 90,
) -> list[Chunk]:
    """Topic-aware: break where consecutive sentences diverge in meaning.

    Embed each sentence, measure cosine distance between neighbors, and cut at
    the distances above ``breakpoint_percentile``. ``embed_fn`` is injectable so
    the boundary logic can be tested without calling OpenAI.
    """
    embed_fn = embed_fn or embed_texts
    source = segments[0].source if segments else ""
    text = "\n\n".join(seg.text for seg in segments)
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [_chunk(text, source, 0, "semantic")] if text else []

    vectors = np.array(embed_fn(sentences))
    distances = _consecutive_distances(vectors)
    threshold = np.percentile(distances, breakpoint_percentile)
    boundaries = [i + 1 for i, d in enumerate(distances) if d > threshold]

    groups = _split_at(sentences, boundaries)
    return [_chunk(" ".join(g), source, i, "semantic") for i, g in enumerate(groups)]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = size - overlap
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _consecutive_distances(vectors: np.ndarray) -> np.ndarray:
    a, b = vectors[:-1], vectors[1:]
    cos = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )
    return 1 - cos


def _split_at(items: list[str], boundaries: list[int]) -> list[list[str]]:
    groups, start = [], 0
    for boundary in boundaries:
        groups.append(items[start:boundary])
        start = boundary
    groups.append(items[start:])
    return [g for g in groups if g]


def _chunk(
    text: str,
    source: str,
    index: int,
    strategy: str,
    heading: str | None = None,
    page: int | None = None,
) -> Chunk:
    return Chunk(
        text=text,
        source=source,
        chunk_index=index,
        strategy=strategy,
        char_count=len(text),
        heading=heading,
        page=page,
    )
