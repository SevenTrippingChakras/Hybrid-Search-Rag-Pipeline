"""Tests for the store abstraction.

These stay offline: the factory is checked by name only, and the in-memory
FakeHybridStore proves the ``Index`` depends on the ``HybridStore`` port, not on
OpenSearch.
"""

import pytest

from app.rag.index import Index
from app.rag.models import Chunk
from app.rag.stores import HybridStore, build_store
from tests.support import FakeHybridStore, fake_embed


def test_build_store_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown vector backend"):
        build_store("cassandra")


def test_fake_store_satisfies_the_port():
    assert isinstance(FakeHybridStore(), HybridStore)


def test_index_works_against_any_store():
    store = FakeHybridStore()
    idx = Index(store=store, embed_fn=fake_embed)
    idx.add([Chunk("a body", "doc.md", 0, "fixed", 6)])

    assert idx.count == 1
    assert store.records["doc.md:fixed:0"][1] == "a body"
    assert store.sparse_query("body", k=1)[0].id == "doc.md:fixed:0"
