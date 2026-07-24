"""Tests for the configurable chunking strategies."""

import pytest
import tiktoken

from app.rag.chunking import chunk, chunk_by_header, chunk_fixed, chunk_semantic
from app.rag.models import Segment

_ENC = tiktoken.get_encoding("cl100k_base")


def _seg(text, heading=None, page=None, source="doc.md"):
    return Segment(text=text, source=source, heading=heading, page=page)


def _tokens(text):
    return len(_ENC.encode(text))


def test_fixed_windows_overlap_and_stamp_strategy():
    # Sizing is token-based: build text of exactly 250 tokens from one token id.
    segments = [_seg(_ENC.decode([828] * 250))]
    chunks = chunk_fixed(segments, chunk_size=100, overlap=20)

    # step = size - overlap = 80 tokens, so windows start at token 0, 80, 160.
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.strategy == "fixed" and c.heading is None for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)
    assert _tokens(chunks[0].text) == 100


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
    # the heading is prepended to its own chunk's text (retrieval signal)
    assert chunks[0].text == "Alpha\nAlpha section body."
    assert chunks[1].text == "Beta\nBeta section body."


def test_header_without_heading_leaves_text_unprepended():
    chunks = chunk_by_header([_seg("Preamble before any heading.")])

    assert chunks[0].heading is None
    assert chunks[0].text == "Preamble before any heading."


def test_header_keeps_atomic_elements_whole_and_tags_them():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(60)) + "\n```"
    body = (
        "Intro prose one. Intro prose two.\n\n"
        f"{code}\n\n"
        "| Col A | Col B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "- first item\n- second item\n- third item\n\n"
        "Closing prose."
    )
    # tiny size forces prose to split, so an intact code block proves protection
    chunks = chunk_by_header(
        [_seg(body, heading="H", page=1)], chunk_size=20, overlap=5
    )

    by_type = [c.element_type for c in chunks]
    assert "code" in by_type and "table" in by_type and "list" in by_type
    assert any(c.element_type is None for c in chunks)  # prose still present

    code_chunk = next(c for c in chunks if c.element_type == "code")
    assert "x0 = 0" in code_chunk.text and "x59 = 59" in code_chunk.text  # whole
    assert _tokens(code_chunk.text) > 20  # exceeds size but was never split
    assert all(c.heading == "H" and c.page == 1 for c in chunks)


def test_header_prose_only_section_has_no_element_type():
    chunks = chunk_by_header([_seg("Just plain prose here.", heading="H")])
    assert chunks[0].element_type is None


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
