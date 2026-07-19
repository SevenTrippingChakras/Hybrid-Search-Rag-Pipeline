"""Cloud vector-store adapters: Pinecone and Milvus (Zilliz Cloud).

Kept out of :mod:`hybrid_rag.stores` so the heavy SDKs load only when the
matching backend is actually selected. Both implement the same ``VectorStore``
port: dense vectors are handed in, the chunk text rides along as a ``text`` field
so ``query`` can return the document, and cosine similarity is used so ``score``
means "higher is more similar" — identical semantics to ``ChromaStore``.

Credentials come from the environment (``PINECONE_*`` / ``MILVUS_*``). Each store
assumes its index/collection is dimensioned for ``text-embedding-3-small``.
"""

from __future__ import annotations

from hybrid_rag.config import settings
from hybrid_rag.stores import EMBED_DIM, QueryHit


class PineconeStore:
    """Pinecone serverless index. Needs ``PINECONE_API_KEY`` + ``PINECONE_INDEX``."""

    def __init__(self, index: str | None = None, api_key: str | None = None) -> None:
        from pinecone import Pinecone

        api_key = api_key or settings.pinecone_api_key
        index = index or settings.pinecone_index
        self._index = Pinecone(api_key=api_key).Index(index)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        vectors = [
            {"id": i, "values": e, "metadata": {**m, "text": d}}
            for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True)
        ]
        self._index.upsert(vectors=vectors)

    def query(self, embedding, k: int = 10) -> list[QueryHit]:
        res = self._index.query(vector=embedding, top_k=k, include_metadata=True)
        hits = []
        for match in res["matches"]:
            meta = dict(match["metadata"])
            doc = meta.pop("text", "")
            hits.append(QueryHit(match["id"], doc, meta, match["score"]))
        return hits

    def count(self) -> int:
        return self._index.describe_index_stats()["total_vector_count"]


class MilvusStore:
    """Milvus / Zilliz Cloud collection. Needs ``MILVUS_URI`` (+ ``MILVUS_TOKEN``)."""

    _FIELDS = [
        "text",
        "source",
        "chunk_index",
        "strategy",
        "char_count",
        "heading",
        "page",
    ]

    def __init__(
        self,
        collection: str = "chunks",
        uri: str | None = None,
        token: str | None = None,
    ) -> None:
        from pymilvus import DataType, MilvusClient

        uri = uri or settings.milvus_uri
        token = token or settings.milvus_token
        self._client = MilvusClient(uri=uri, token=token)
        self._collection = collection
        if not self._client.has_collection(collection):
            schema = self._client.create_schema(
                auto_id=False, enable_dynamic_field=True
            )
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=512)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self._client.create_collection(
                collection, schema=schema, index_params=index_params
            )
        self._client.load_collection(collection)

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        data = [
            {"id": i, "vector": e, "text": d, **m}
            for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, data=data)

    def query(self, embedding, k: int = 10) -> list[QueryHit]:
        res = self._client.search(
            collection_name=self._collection,
            data=[embedding],
            limit=k,
            output_fields=self._FIELDS,
        )
        hits = []
        for match in res[0]:
            entity = dict(match["entity"])
            doc = entity.pop("text", "")
            hits.append(QueryHit(str(match["id"]), doc, entity, match["distance"]))
        return hits

    def count(self) -> int:
        self._client.flush(self._collection)
        return self._client.get_collection_stats(self._collection)["row_count"]
