"""The retrieval store behind one interface.

``HybridStore`` is the port; ``OpenSearchStore`` is the adapter, selected by
``build_store`` from the ``VECTOR_BACKEND`` env var. One engine does both dense
k-NN (cosine ``score``, higher is closer) and sparse BM25 over the same index.
The factory stays so a future adapter (e.g. Azure AI Search) remains swappable.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.config import settings

EMBED_DIM = 1536  # text-embedding-3-small


@dataclass
class QueryHit:
    """One search result: the stored chunk, its metadata, and a similarity score."""

    id: str
    document: str
    metadata: dict
    score: float


@runtime_checkable
class HybridStore(Protocol):
    """The retrieval port: write once, read both dense and sparse."""

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    def query(self, embedding: list[float], k: int = 10) -> list[QueryHit]: ...

    def sparse_query(self, text: str, k: int = 10) -> list[QueryHit]: ...

    def hybrid_query(
        self, text: str, embedding: list[float], k: int = 10
    ) -> list[QueryHit]: ...

    def count(self) -> int: ...


class OpenSearchStore:
    """OpenSearch: dense k-NN and sparse BM25 over one index.

    A single document per chunk carries both a ``knn_vector`` embedding (cosine)
    and the analyzed ``text`` field, so ``query`` (dense) and ``sparse_query``
    (BM25) read the same server-side, replica-safe index. Retires the Chroma +
    local BM25-sidecar split.
    """

    _PIPELINE = "rrf-pipeline"

    def __init__(self, host: str | None = None, index: str | None = None) -> None:
        from opensearchpy import OpenSearch

        self._index = index or settings.opensearch_index
        self._client = OpenSearch(hosts=[host or settings.opensearch_host])
        self._ensure_index()
        self._ensure_pipeline()

    def _ensure_pipeline(self) -> None:
        """Server-side RRF fusion pipeline for the native hybrid query (idempotent)."""
        self._client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{self._PIPELINE}",
            body={
                "description": "hybrid RRF fusion",
                "phase_results_processors": [
                    {
                        "score-ranker-processor": {
                            "combination": {
                                "technique": "rrf",
                                "rank_constant": settings.rrf_k,
                            }
                        }
                    }
                ],
            },
        )

    def _ensure_index(self) -> None:
        if self._client.indices.exists(index=self._index):
            return
        self._client.indices.create(
            index=self._index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": EMBED_DIM,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "lucene",
                            },
                        },
                        "metadata": {"type": "object"},
                    }
                },
            },
        )

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        from opensearchpy.helpers import bulk

        actions = [
            {
                "_index": self._index,
                "_id": id_,
                "_source": {"text": doc, "embedding": emb, "metadata": meta},
            }
            for id_, emb, doc, meta in zip(
                ids, embeddings, documents, metadatas, strict=True
            )
        ]
        bulk(self._client, actions)
        self._client.indices.refresh(index=self._index)

    # Reads never need the stored vector back; excluding it trims the response.
    _SOURCE = {"excludes": ["embedding"]}

    def query(self, embedding, k: int = 10) -> list[QueryHit]:
        """Dense k-NN. Converts OpenSearch's cosinesimil score back to cosine."""
        res = self._client.search(
            index=self._index,
            body={
                "size": k,
                "_source": self._SOURCE,
                "query": {"knn": {"embedding": {"vector": embedding, "k": k}}},
            },
        )
        return [self._hit(h, cosine=True) for h in res["hits"]["hits"]]

    def sparse_query(self, text: str, k: int = 10) -> list[QueryHit]:
        """BM25 keyword match. Returns Lucene's relevance score as-is."""
        res = self._client.search(
            index=self._index,
            body={
                "size": k,
                "_source": self._SOURCE,
                "query": {"match": {"text": text}},
            },
        )
        return [self._hit(h, cosine=False) for h in res["hits"]["hits"]]

    def hybrid_query(
        self, text: str, embedding: list[float], k: int = 10
    ) -> list[QueryHit]:
        """Native hybrid: one request, BM25 + k-NN fused server-side by RRF.

        The ``score`` is the RRF fusion score, not cosine.
        """
        res = self._client.search(
            index=self._index,
            body={
                "size": k,
                "_source": self._SOURCE,
                "query": {
                    "hybrid": {
                        "queries": [
                            {"match": {"text": {"query": text}}},
                            {"knn": {"embedding": {"vector": embedding, "k": k}}},
                        ]
                    }
                },
            },
            params={"search_pipeline": self._PIPELINE},
        )
        return [self._hit(h, cosine=False) for h in res["hits"]["hits"]]

    def count(self) -> int:
        self._client.indices.refresh(index=self._index)
        return self._client.count(index=self._index)["count"]

    @staticmethod
    def _hit(h: dict, cosine: bool) -> QueryHit:
        src = h["_source"]
        score = 2.0 * h["_score"] - 1.0 if cosine else h["_score"]
        return QueryHit(h["_id"], src["text"], dict(src.get("metadata", {})), score)


def build_store(backend: str | None = None, index: str | None = None) -> HybridStore:
    """Construct the store named by ``VECTOR_BACKEND`` (default ``opensearch``).

    ``index`` overrides the default index name, so callers can target a
    per-strategy index (see :func:`index_for_strategy`).
    """
    backend = (backend or settings.vector_backend).lower()
    if backend == "opensearch":
        return OpenSearchStore(index=index)
    raise ValueError(f"Unknown vector backend: {backend!r}")


def index_for_strategy(strategy: str) -> str:
    """The per-strategy index name, e.g. ``chunks_fixed``.

    Each chunking strategy lives in its own index so the three coexist and can be
    evaluated without wiping between runs.
    """
    return f"{settings.opensearch_index}_{strategy}"
