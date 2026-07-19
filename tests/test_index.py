"""Tests for the indexing write path (dense + sparse kept in sync).

An injected fake embedder, a ChromaStore under tmp_path, and a tmp sparse sidecar
keep the suite offline and isolated from the real ``data/index``.
"""

import hashlib

from hybrid_rag.index import Index, _chunk_id
from hybrid_rag.models import Chunk
from hybrid_rag.sparse import Bm25Store
from hybrid_rag.stores import ChromaStore

_EMBED_DIM = 32


def _fake_embed(texts):
    """Deterministic bag-of-words vectors, no network.

    Identical text yields an identical vector (cosine 1.0, so dedup catches it);
    different words hash to different buckets, so distinct chunks stay dissimilar.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * _EMBED_DIM
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _EMBED_DIM
            vec[bucket] += 1.0
        vectors.append(vec)
    return vectors


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
    sparse = Bm25Store(path=str(tmp_path / "index" / "bm25.json"))
    return Index(store=store, sparse=sparse, embed_fn=_fake_embed)


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


def test_sparse_store_stays_in_sync_with_dense(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("the cat sat on the mat", 0)])
    idx.add([_chunk("the dog chased the ball", 1)])
    idx.add([_chunk("the engine needs oil", 2)])

    # both stores hold the same chunks
    assert idx.count == 3
    assert idx._sparse.count() == 3
    # and the sparse store can find them by keyword
    top = idx._sparse.query("engine oil", k=1)[0]
    assert top.id == _chunk_id(_chunk("the engine needs oil", 2))


def test_add_empty_is_noop(tmp_path):
    idx = _index(tmp_path)
    idx.add([])
    assert idx.count == 0
    assert idx._sparse.count() == 0


def test_near_duplicate_across_docs_is_skipped(tmp_path):
    idx = _index(tmp_path)
    idx.add([_chunk("shared policy text", 0, source="a.md")])
    result = idx.add([_chunk("shared policy text", 0, source="b.md")])

    assert idx.count == 1
    assert result.added == []
    assert result.skipped == [_chunk_id(_chunk("shared policy text", 0, source="b.md"))]


def test_duplicates_within_a_batch_are_skipped(tmp_path):
    idx = _index(tmp_path)
    result = idx.add(
        [
            _chunk("repeated body", 0, source="a.md"),
            _chunk("repeated body", 0, source="b.md"),
        ]
    )

    assert idx.count == 1
    assert len(result.added) == 1
    assert len(result.skipped) == 1


def test_distinct_chunks_are_all_kept(tmp_path):
    idx = _index(tmp_path)
    result = idx.add(
        [_chunk("alpha content here", 0), _chunk("totally different words", 1)]
    )

    assert idx.count == 2
    assert len(result.added) == 2
    assert result.skipped == []


def test_dedup_disabled_keeps_duplicates(tmp_path):
    store = ChromaStore(path=str(tmp_path / "index"))
    sparse = Bm25Store(path=str(tmp_path / "index" / "bm25.json"))
    idx = Index(
        store=store,
        sparse=sparse,
        embed_fn=_fake_embed,
        dedup_threshold=1.0,
    )
    idx.add([_chunk("shared policy text", 0, source="a.md")])
    idx.add([_chunk("shared policy text", 0, source="b.md")])

    assert idx.count == 2


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
