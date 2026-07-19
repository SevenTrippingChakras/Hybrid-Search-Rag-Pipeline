"""Tests for the indexing write path.

An injected fake embedder and an in-memory FakeHybridStore keep the suite offline
and isolated from a live OpenSearch.
"""

from hybrid_rag.index import Index, _chunk_id
from hybrid_rag.models import Chunk
from tests.support import FakeHybridStore, fake_embed


def _chunk(text, index, source="doc.md", strategy="fixed", heading=None, page=None):
    return Chunk(
        text=text,
        source=source,
        chunk_index=index,
        strategy=strategy,
        char_count=len(text),
        heading=heading,
        page=page,
    )


def _index():
    return Index(store=FakeHybridStore(), embed_fn=fake_embed)


def test_add_stores_chunks_with_metadata():
    idx = _index()
    idx.add([_chunk("hello world", 0, heading="Intro", page=3)])

    assert idx.count == 1
    _embedding, document, meta = idx._store.records["doc.md:fixed:0"]
    assert document == "hello world"
    assert meta["source"] == "doc.md"
    assert meta["chunk_index"] == 0
    assert meta["strategy"] == "fixed"
    assert meta["char_count"] == 11
    assert meta["heading"] == "Intro"
    assert meta["page"] == 3


def test_none_metadata_fields_are_dropped():
    idx = _index()
    idx.add([_chunk("no structure", 0)])

    _embedding, _document, meta = idx._store.records["doc.md:fixed:0"]
    assert "heading" not in meta
    assert "page" not in meta


def test_reindexing_same_chunk_upserts():
    idx = _index()
    idx.add([_chunk("first", 0)])
    idx.add([_chunk("first revised", 0)])

    assert idx.count == 1
    assert idx._store.records["doc.md:fixed:0"][1] == "first revised"


def test_sparse_side_finds_chunks_by_keyword():
    idx = _index()
    idx.add([_chunk("the cat sat on the mat", 0)])
    idx.add([_chunk("the dog chased the ball", 1)])
    idx.add([_chunk("the engine needs oil", 2)])

    assert idx.count == 3
    top = idx._store.sparse_query("engine oil", k=1)[0]
    assert top.id == _chunk_id(_chunk("the engine needs oil", 2))


def test_add_empty_is_noop():
    idx = _index()
    idx.add([])
    assert idx.count == 0


def test_near_duplicate_across_docs_is_skipped():
    idx = _index()
    idx.add([_chunk("shared policy text", 0, source="a.md")])
    result = idx.add([_chunk("shared policy text", 0, source="b.md")])

    assert idx.count == 1
    assert result.added == []
    assert result.skipped == [_chunk_id(_chunk("shared policy text", 0, source="b.md"))]


def test_duplicates_within_a_batch_are_skipped():
    idx = _index()
    result = idx.add(
        [
            _chunk("repeated body", 0, source="a.md"),
            _chunk("repeated body", 0, source="b.md"),
        ]
    )

    assert idx.count == 1
    assert len(result.added) == 1
    assert len(result.skipped) == 1


def test_distinct_chunks_are_all_kept():
    idx = _index()
    result = idx.add(
        [_chunk("alpha content here", 0), _chunk("totally different words", 1)]
    )

    assert idx.count == 2
    assert len(result.added) == 2
    assert result.skipped == []


def test_dedup_disabled_keeps_duplicates():
    idx = Index(store=FakeHybridStore(), embed_fn=fake_embed, dedup_threshold=1.0)
    idx.add([_chunk("shared policy text", 0, source="a.md")])
    idx.add([_chunk("shared policy text", 0, source="b.md")])

    assert idx.count == 2
