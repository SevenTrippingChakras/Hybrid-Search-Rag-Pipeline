"""Offline test doubles shared across the suite.

``FakeHybridStore`` is an in-memory ``HybridStore``: real cosine for dense
queries, token-overlap for sparse, so Index/Retriever logic is tested without a
live OpenSearch. ``fake_embed`` gives deterministic bag-of-words vectors.
"""

import hashlib
import math

from hybrid_rag.stores import QueryHit

EMBED_DIM = 32


def fake_embed(texts):
    """Deterministic bag-of-words vectors, no network.

    Identical text yields an identical vector (cosine 1.0, so dedup catches it);
    different words hash to different buckets, so distinct chunks stay dissimilar.
    """
    vectors = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBED_DIM
            vec[bucket] += 1.0
        vectors.append(vec)
    return vectors


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class FakeHybridStore:
    """In-memory hybrid store: cosine dense search, token-overlap sparse search."""

    def __init__(self):
        self.records: dict[str, tuple] = {}  # id -> (embedding, document, metadata)

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            self.records[i] = (e, d, m)

    def query(self, embedding, k=10):
        scored = [
            (i, _cosine(embedding, e), d, m) for i, (e, d, m) in self.records.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [QueryHit(i, d, m, s) for i, s, d, m in scored[:k]]

    def sparse_query(self, text, k=10):
        q = set(text.lower().split())
        scored = []
        for i, (_e, d, m) in self.records.items():
            overlap = sum(1 for w in d.lower().split() if w in q)
            if overlap:
                scored.append((i, float(overlap), d, m))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [QueryHit(i, d, m, s) for i, s, d, m in scored[:k]]

    def hybrid_query(self, text, embedding, k=10, rank_constant=60):
        """Mimics OpenSearch native RRF: rank-fuse dense and sparse results."""
        n = len(self.records)
        scores: dict[str, float] = {}
        hits: dict[str, QueryHit] = {}
        for ranked in (self.query(embedding, k=n), self.sparse_query(text, k=n)):
            for rank, hit in enumerate(ranked):
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rank_constant + rank)
                hits[hit.id] = hit
        top = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:k]
        return [
            QueryHit(hits[i].id, hits[i].document, hits[i].metadata, s) for i, s in top
        ]

    def count(self):
        return len(self.records)


class FakeStorage:
    """In-memory object storage: proves callers depend on the StorageBackend port."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put(self, key, data, content_type=None):
        self.objects[key] = data

    def get(self, key):
        return self.objects[key]

    def delete(self, key):
        self.objects.pop(key, None)

    def exists(self, key):
        return key in self.objects
