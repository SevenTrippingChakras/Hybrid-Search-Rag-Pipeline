"""Tests for the BM25 sparse store."""

from hybrid_rag.sparse import Bm25Store, SparseStore, _tokenize


def _store(tmp_path):
    return Bm25Store(path=str(tmp_path / "bm25.json"))


def test_bm25_store_satisfies_the_port(tmp_path):
    assert isinstance(_store(tmp_path), SparseStore)


def test_query_ranks_keyword_matches_first(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=["a", "b", "c"],
        documents=[
            "the cat sat on the mat",
            "the dog chased the ball",
            "the engine needs oil",
        ],
        metadatas=[{}, {}, {}],
    )
    hits = store.query("engine oil", k=3)

    assert hits[0].id == "c"
    assert hits[0].score > hits[1].score


def test_query_returns_document_and_metadata(tmp_path):
    store = _store(tmp_path)
    store.add(ids=["a"], documents=["engine oil"], metadatas=[{"source": "x.md"}])
    hit = store.query("engine", k=1)[0]

    assert hit.document == "engine oil"
    assert hit.metadata["source"] == "x.md"


def test_query_respects_k(tmp_path):
    store = _store(tmp_path)
    store.add(
        ids=[str(n) for n in range(5)],
        documents=[f"word number {n}" for n in range(5)],
        metadatas=[{}] * 5,
    )
    assert len(store.query("word", k=3)) == 3


def test_empty_store_query_returns_nothing(tmp_path):
    assert _store(tmp_path).query("anything") == []


def test_reindexing_same_id_upserts(tmp_path):
    store = _store(tmp_path)
    store.add(ids=["a"], documents=["first"], metadatas=[{}])
    store.add(ids=["a"], documents=["first revised"], metadatas=[{}])

    assert store.count() == 1
    assert store.query("revised", k=1)[0].document == "first revised"


def test_corpus_persists_across_reopen(tmp_path):
    store = _store(tmp_path)
    store.add(ids=["a"], documents=["persisted body text"], metadatas=[{}])

    reopened = _store(tmp_path)
    assert reopened.count() == 1
    assert reopened.query("persisted", k=1)[0].id == "a"


def test_tokenize_lowercases_and_splits_on_word_chars():
    assert _tokenize("Hello, WORLD! foo_bar") == ["hello", "world", "foo_bar"]
