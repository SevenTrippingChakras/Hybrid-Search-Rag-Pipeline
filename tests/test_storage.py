"""Tests for the object storage abstraction.

These stay offline: the factory is checked by name only, and the in-memory
FakeStorage proves callers depend on the ``StorageBackend`` port, not on boto3.
A live round-trip against a real bucket is scripts/check_storage.py instead.
"""

import pytest

from hybrid_rag.storage import StorageBackend, build_storage
from tests.support import FakeStorage


def test_build_storage_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown storage backend"):
        build_storage("dropbox")


def test_fake_storage_satisfies_the_port():
    assert isinstance(FakeStorage(), StorageBackend)


def test_fake_storage_round_trips():
    storage = FakeStorage()
    storage.put("docs/a.txt", b"hello")

    assert storage.exists("docs/a.txt")
    assert storage.get("docs/a.txt") == b"hello"

    storage.delete("docs/a.txt")
    assert not storage.exists("docs/a.txt")
