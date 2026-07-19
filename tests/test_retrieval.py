"""Tests for the retrieval read path.

The Retriever reads the same stores the Index writes. Here it is wired to a
ChromaStore + Bm25Store under tmp_path, populated through an Index with a fake
embedder, so the suite stays offline.
"""

import hashlib

from hybrid_rag.index import Index
from hybrid_rag.models import Chunk
from hybrid_rag.retrieval import Retriever
from hybrid_rag.sparse import Bm25Store
from hybrid_rag.stores import ChromaStore

_EMBED_DIM = 32


def _fake_embed(texts):
    """Deterministic bag-of-words vectors, no network (matches test_index)."""
    vectors = []
    for text in texts:
        vec = [0.0] * _EMBED_DIM
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _EMBED_DIM
            vec[bucket] += 1.0
        vectors.append(vec)
    return vectors


def _chunk(text, index):
    return Chunk(
        text=text,
        source="doc.md",
        chunk_index=index,
        strategy="fixed",
        char_count=len(text),
    )


def _index_and_retriever(tmp_path):
    """Index and Retriever sharing the same dense + sparse store instances."""
    store = ChromaStore(path=str(tmp_path / "index"))
    sparse = Bm25Store(path=str(tmp_path / "index" / "bm25.json"))
    idx = Index(store=store, sparse=sparse, embed_fn=_fake_embed)
    retriever = Retriever(store=store, sparse=sparse, embed_fn=_fake_embed)
    return idx, retriever


def test_dense_search_ranks_by_similarity(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk("the engine needs oil", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])

    hits = retriever.dense_search("engine oil", k=2)

    assert len(hits) == 2
    assert hits[0].document == "the engine needs oil"
    assert hits[0].score > hits[1].score


def test_dense_search_returns_metadata(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk("engine oil", 0)])

    hit = retriever.dense_search("engine", k=1)[0]
    assert hit.metadata["source"] == "doc.md"
    assert hit.metadata["strategy"] == "fixed"


def test_dense_search_respects_k(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.dense_search("words", k=3)) == 3


def test_dense_search_empty_index_returns_nothing(tmp_path):
    _idx, retriever = _index_and_retriever(tmp_path)
    assert retriever.dense_search("anything") == []


def test_sparse_search_ranks_by_keyword_overlap(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("a dog ran across the yard", 2)])
    idx.add([_chunk("birds fly south for winter", 3)])

    hits = retriever.sparse_search("E42 error code", k=4)

    assert hits[0].document == "restart the engine to clear error code E42"
    assert hits[0].score > hits[1].score


def test_sparse_search_returns_metadata(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk("engine oil", 0)])

    hit = retriever.sparse_search("engine", k=1)[0]
    assert hit.metadata["source"] == "doc.md"
    assert hit.metadata["strategy"] == "fixed"


def test_sparse_search_respects_k(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.sparse_search("words", k=3)) == 3


def test_sparse_search_empty_index_returns_nothing(tmp_path):
    _idx, retriever = _index_and_retriever(tmp_path)
    assert retriever.sparse_search("anything") == []


def test_hybrid_search_fuses_dense_and_sparse(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("a dog ran across the yard", 2)])

    hits = retriever.hybrid_search("engine error code E42", k=3)

    assert hits[0].document == "restart the engine to clear error code E42"
    assert hits[0].score > hits[1].score


def test_hybrid_search_respects_k(tmp_path):
    idx, retriever = _index_and_retriever(tmp_path)
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.hybrid_search("words", k=2)) == 2


def test_hybrid_search_empty_index_returns_nothing(tmp_path):
    _idx, retriever = _index_and_retriever(tmp_path)
    assert retriever.hybrid_search("anything") == []


class _FakeReranker:
    """Reranks by query-token overlap; proves Retriever depends on the port."""

    def rerank(self, query, hits, top_k=5):
        q = set(query.lower().split())
        scored = sorted(
            hits,
            key=lambda h: sum(1 for w in h.document.lower().split() if w in q),
            reverse=True,
        )
        return scored[:top_k]


def test_search_reranks_hybrid_candidates(tmp_path):
    store = ChromaStore(path=str(tmp_path / "index"))
    sparse = Bm25Store(path=str(tmp_path / "index" / "bm25.json"))
    idx = Index(store=store, sparse=sparse, embed_fn=_fake_embed)
    retriever = Retriever(
        store=store, sparse=sparse, reranker=_FakeReranker(), embed_fn=_fake_embed
    )
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("a dog ran across the yard", 2)])

    hits = retriever.search("engine error code E42", top_k=2)

    assert len(hits) == 2
    assert hits[0].document == "restart the engine to clear error code E42"
