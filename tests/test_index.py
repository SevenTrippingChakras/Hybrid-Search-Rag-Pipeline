"""Tests for the dense + sparse index.

An injected fake embedder, a ChromaStore under tmp_path, and a tmp sparse sidecar
keep the suite offline and isolated from the real ``data/index``.
"""

from hybrid_rag.index import Index, _chunk_id, _tokenize
from hybrid_rag.models import Chunk
from hybrid_rag.stores import ChromaStore


def _fake_embed(texts):
    """Deterministic 2-d vectors, no network."""
    return [[float(len(t)), 1.0] for t in texts]


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


def _index(tmp_path):
    store = ChromaStore(path=str(tmp_path / "index"))
    return Index(
        store=store,
        sparse_path=str(tmp_path / "index" / "bm25.json"),
        embed_fn=_fake_embed,
    )


def test_add_stores_chunks_with_metadata(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("hello world", 0, heading="Intro", page=3)])

    assert idx.count == 1
    stored = idx._store._collection.get(include=["metadatas", "documents"])
    assert stored["documents"] == ["hello world"]
    meta = stored["metadatas"][0]
    assert meta["source"] == "doc.md"
    assert meta["chunk_index"] == 0
    assert meta["strategy"] == "fixed"
    assert meta["char_count"] == 11
    assert meta["heading"] == "Intro"
    assert meta["page"] == 3


def test_none_metadata_fields_are_dropped(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("no structure", 0)])

    meta = idx._store._collection.get(include=["metadatas"])["metadatas"][0]
    assert "heading" not in meta
    assert "page" not in meta


def test_reindexing_same_chunk_upserts(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("first", 0)])
    idx.add([_chunk("first revised", 0)])

    assert idx.count == 1
    docs = idx._store._collection.get(include=["documents"])["documents"]
    assert docs == ["first revised"]


def test_sparse_index_stays_in_sync_with_dense(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("the cat sat on the mat", 0)])
    idx.add([_chunk("the dog chased the ball", 1)])
    idx.add([_chunk("the engine needs oil", 2)])

    assert idx.count == 3
    assert idx._bm25 is not None
    assert len(idx._ids) == 3
    # BM25 corpus size tracks the dense store.
    scores = idx._bm25.get_scores(_tokenize("engine oil"))
    assert len(scores) == 3
    # the engine chunk outscores the cat chunk on an engine query
    engine_pos = idx._ids.index(_chunk_id(_chunk("the engine needs oil", 2)))
    cat_pos = idx._ids.index(_chunk_id(_chunk("the cat sat on the mat", 0)))
    assert scores[engine_pos] > scores[cat_pos]


def test_sparse_corpus_persists_across_reopen(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("persisted body text", 0)])

    reopened = _index(tmp_path)
    assert reopened._ids == [_chunk_id(_chunk("persisted body text", 0))]
    assert reopened._bm25 is not None


def test_add_empty_is_noop(tmp_path):
    idx = _index(tmp_path)
    idx.add([])
    assert idx.count == 0
    assert idx._bm25 is None


def test_chroma_store_query_returns_scored_hits(tmp_path):
    store = ChromaStore(path=str(tmp_path / "index"))
    store.upsert(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["alpha doc", "beta doc"],
        metadatas=[{"source": "x"}, {"source": "y"}],
    )
    hits = store.query([1.0, 0.0], k=2)

    assert [h.id for h in hits] == ["a", "b"]  # nearest first
    assert hits[0].document == "alpha doc"
    assert hits[0].metadata["source"] == "x"
    # cosine similarity of identical direction is ~1.0
    assert hits[0].score > hits[1].score
