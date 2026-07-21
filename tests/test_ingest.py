"""Tests for the full-corpus ingest driver.

A FakeHybridStore and a fake embedder keep the suite offline: no OpenAI, no live
OpenSearch. Files are written to a tmp corpus so discovery and indexing are
exercised end to end.
"""

from hybrid_rag.index import Index
from hybrid_rag.ingest import find_documents, ingest_corpus
from tests.support import FakeHybridStore, fake_embed


def _index():
    return Index(store=FakeHybridStore(), embed_fn=fake_embed)


def _write(dir_, name, text):
    path = dir_ / name
    path.write_text(text, encoding="utf-8")
    return path


def test_find_documents_picks_supported_formats_only(tmp_path):
    _write(tmp_path, "a.md", "# Title\nbody")
    _write(tmp_path, "b.txt", "plain text")
    _write(tmp_path, "c.html", "<p>html body</p>")
    _write(tmp_path, "skip.json", "{}")

    found = {p.name for p in find_documents(tmp_path)}

    assert found == {"a.md", "b.txt", "c.html"}
    assert "skip.json" not in found


def test_find_documents_is_recursive(tmp_path):
    nested = tmp_path / "sub"
    nested.mkdir()
    _write(nested, "deep.md", "# Deep\nnested body")

    found = [p.name for p in find_documents(tmp_path)]

    assert "deep.md" in found


def test_ingest_corpus_indexes_all_documents(tmp_path):
    _write(tmp_path, "one.md", "# One\nThe first document has some words.")
    _write(tmp_path, "two.txt", "The second document has different words entirely.")

    report = ingest_corpus(tmp_path, index=_index())

    assert report.files == 2
    assert report.chunks_added >= 2
    assert report.chunks_skipped == 0


def test_ingest_corpus_limit_stops_early(tmp_path):
    for i in range(5):
        _write(tmp_path, f"doc_{i}.md", f"# Doc {i}\nUnique body number {i} here.")

    report = ingest_corpus(tmp_path, limit=2, index=_index())

    assert report.files == 2
