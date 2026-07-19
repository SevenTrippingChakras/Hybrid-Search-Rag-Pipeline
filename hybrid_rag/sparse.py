"""Sparse (BM25 keyword) store behind a small port.

The dense side has a swappable ``VectorStore`` (:mod:`hybrid_rag.stores`); this
is its sparse counterpart. ``SparseStore`` is the port, ``Bm25Store`` the only
adapter for now — a local BM25 index over a JSON corpus sidecar. Keeping sparse
behind a port, symmetric with the dense store, lets the indexer write both and
the retriever read both without either depending on BM25 specifics. Phase 2.5
retires this file when OpenSearch absorbs dense and sparse into one engine.

BM25 matches exact keywords — function names, config keys, error codes — that
semantic search can miss, so it complements dense retrieval rather than
replacing it. Each entry keeps its text and metadata (like a dense hit) so a
search result carries its own provenance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from rank_bm25 import BM25Okapi

from hybrid_rag.stores import QueryHit

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class SparseStore(Protocol):
    """The sparse-store port: same add/query/count shape as the dense store."""

    def add(
        self, ids: list[str], documents: list[str], metadatas: list[dict]
    ) -> None: ...

    def query(self, text: str, k: int = 10) -> list[QueryHit]: ...

    def count(self) -> int: ...


class Bm25Store:
    """A local BM25 index persisted as a JSON corpus sidecar.

    The BM25 index is held in memory and rebuilt from the corpus on every ``add``
    and on load; stable ids make re-adding a chunk an upsert. ``query`` returns
    ``QueryHit`` objects (the same type the dense store returns) so the retriever
    can fuse dense and sparse results uniformly.
    """

    def __init__(self, path: str = "data/index/bm25.json") -> None:
        self._path = Path(path)
        self._corpus: dict[str, dict] = self._load()
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild()

    def add(self, ids, documents, metadatas) -> None:
        for id_, doc, meta in zip(ids, documents, metadatas, strict=True):
            self._corpus[id_] = {"text": doc, "metadata": meta}
        self._save()
        self._rebuild()

    def query(self, text: str, k: int = 10) -> list[QueryHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(
            zip(self._ids, scores, strict=True), key=lambda p: p[1], reverse=True
        )
        hits = []
        for id_, score in ranked[:k]:
            entry = self._corpus[id_]
            hits.append(
                QueryHit(id_, entry["text"], dict(entry["metadata"]), float(score))
            )
        return hits

    def count(self) -> int:
        return len(self._corpus)

    def _rebuild(self) -> None:
        """Rebuild the in-memory BM25 index from the current corpus."""
        self._ids = list(self._corpus)
        tokens = [_tokenize(self._corpus[i]["text"]) for i in self._ids]
        self._bm25 = BM25Okapi(tokens) if tokens else None

    def _load(self) -> dict[str, dict]:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._corpus, ensure_ascii=False), encoding="utf-8"
        )
