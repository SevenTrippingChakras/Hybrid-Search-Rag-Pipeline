"""Tests for the configurable chunking strategies."""

import pytest

from app.rag.chunking import chunk, chunk_by_header, chunk_fixed, chunk_semantic
from app.rag.models import Segment


def _seg(text, heading=None, page=None, source="doc.md"):
    return Segment(text=text, source=source, heading=heading, page=page)


def test_fixed_windows_overlap_and_stamp_strategy():
    segments = [_seg("A" * 250)]
    chunks = chunk_fixed(segments, chunk_size=100, overlap=20)

    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.strategy == "fixed" and c.heading is None for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)
    # step = size - overlap = 80, so chunk 1 starts 80 chars in and overlaps.
    assert chunks[0].char_count == 100


def test_fixed_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_fixed([_seg("hello world")], chunk_size=50, overlap=50)


def test_header_preserves_heading_and_never_crosses_sections():
    segments = [
        _seg("Alpha section body.", heading="Alpha", page=2),
        _seg("Beta section body.", heading="Beta", page=3),
    ]
    chunks = chunk_by_header(segments)

    assert [c.heading for c in chunks] == ["Alpha", "Beta"]
    assert [c.page for c in chunks] == [2, 3]
    assert all(c.strategy == "header" for c in chunks)
    # each short section is exactly one chunk; no merging across headings
    assert "Alpha" not in chunks[1].text and "Beta" not in chunks[0].text


def test_semantic_cuts_at_topic_boundary_with_injected_embedder():
    def fake_embed(sentences):
        return [[1.0, 0.0] if "cat" in s.lower() else [0.0, 1.0] for s in sentences]

    text = (
        "The cat sat on the mat. The cat drank milk. "
        "The engine needs oil. The engine roared loudly."
    )
    chunks = chunk_semantic([_seg(text, source="pets.txt")], embed_fn=fake_embed)

    assert len(chunks) == 2
    assert "cat" in chunks[0].text and "engine" in chunks[1].text
    assert all(c.strategy == "semantic" for c in chunks)


def test_semantic_single_sentence_is_one_chunk():
    calls = []

    def spy_embed(sentences):
        calls.append(sentences)
        return [[1.0, 0.0] for _ in sentences]

    chunks = chunk_semantic([_seg("Only one sentence here")], embed_fn=spy_embed)

    assert len(chunks) == 1
    assert calls == []  # short-circuits before embedding


def test_dispatch_routes_by_name_and_rejects_unknown():
    segments = [_seg("Some body text.", heading="H")]
    assert chunk(segments, strategy="header")[0].strategy == "header"
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk(segments, strategy="bogus")
