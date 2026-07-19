"""Tests for the retrieval read path.

The Retriever reads the same store the Index writes. Both share one in-memory
FakeHybridStore, populated through an Index with a fake embedder, so the suite
stays offline.
"""

from hybrid_rag.index import Index
from hybrid_rag.models import Chunk
from hybrid_rag.retrieval import Retriever
from tests.support import FakeHybridStore, fake_embed


def _chunk(text, index):
    return Chunk(
        text=text,
        source="doc.md",
        chunk_index=index,
        strategy="fixed",
        char_count=len(text),
    )


def _index_and_retriever():
    """Index and Retriever sharing the same store instance."""
    store = FakeHybridStore()
    idx = Index(store=store, embed_fn=fake_embed)
    retriever = Retriever(store=store, embed_fn=fake_embed)
    return idx, retriever


def test_dense_search_ranks_by_similarity():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk("the engine needs oil", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])

    hits = retriever.dense_search("engine oil", k=2)

    assert len(hits) == 2
    assert hits[0].document == "the engine needs oil"
    assert hits[0].score > hits[1].score


def test_dense_search_returns_metadata():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk("engine oil", 0)])

    hit = retriever.dense_search("engine", k=1)[0]
    assert hit.metadata["source"] == "doc.md"
    assert hit.metadata["strategy"] == "fixed"


def test_dense_search_respects_k():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.dense_search("words", k=3)) == 3


def test_dense_search_empty_index_returns_nothing():
    _idx, retriever = _index_and_retriever()
    assert retriever.dense_search("anything") == []


def test_sparse_search_ranks_by_keyword_overlap():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("the error code compiles cleanly", 2)])
    idx.add([_chunk("birds fly south for winter", 3)])

    hits = retriever.sparse_search("E42 error code", k=4)

    # only docs sharing a term come back; the 3-term match outranks the 2-term one
    assert hits[0].document == "restart the engine to clear error code E42"
    assert hits[0].score > hits[1].score


def test_sparse_search_returns_metadata():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk("engine oil", 0)])

    hit = retriever.sparse_search("engine", k=1)[0]
    assert hit.metadata["source"] == "doc.md"
    assert hit.metadata["strategy"] == "fixed"


def test_sparse_search_respects_k():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.sparse_search("words", k=3)) == 3


def test_sparse_search_empty_index_returns_nothing():
    _idx, retriever = _index_and_retriever()
    assert retriever.sparse_search("anything") == []


def test_hybrid_search_fuses_dense_and_sparse():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("a dog ran across the yard", 2)])

    hits = retriever.hybrid_search("engine error code E42", k=3)

    assert hits[0].document == "restart the engine to clear error code E42"
    assert hits[0].score > hits[1].score


def test_hybrid_search_respects_k():
    idx, retriever = _index_and_retriever()
    idx.add([_chunk(f"unique words number {n}", n) for n in range(5)])

    assert len(retriever.hybrid_search("words", k=2)) == 2


def test_hybrid_search_empty_index_returns_nothing():
    _idx, retriever = _index_and_retriever()
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


def test_search_reranks_hybrid_candidates():
    store = FakeHybridStore()
    idx = Index(store=store, embed_fn=fake_embed)
    retriever = Retriever(store=store, reranker=_FakeReranker(), embed_fn=fake_embed)
    idx.add([_chunk("restart the engine to clear error code E42", 0)])
    idx.add([_chunk("the cat sat on the mat", 1)])
    idx.add([_chunk("a dog ran across the yard", 2)])

    hits = retriever.search("engine error code E42", top_k=2)

    assert len(hits) == 2
    assert hits[0].document == "restart the engine to clear error code E42"
