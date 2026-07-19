"""Central configuration for the pipeline.

One ``Settings`` class reads every knob from the environment (and ``.env``).
``settings`` is a shared instance; import it where you need a value. Optional
fields default to ``None`` so the module imports without a backend's credentials.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None

    # Which dense backend the Index uses: chroma | pinecone | milvus.
    vector_backend: str = "chroma"

    # Deduplication (Phase 1.4): skip a chunk whose cosine similarity to an
    # existing chunk exceeds this. Set >= 1.0 to disable dedup entirely.
    dedup_threshold: float = 0.95

    # Hybrid fusion (Phase 2.3): Reciprocal Rank Fusion weights per list and the
    # RRF rank constant. Higher dense_weight favors semantic hits, higher
    # sparse_weight favors exact keyword hits. rrf_k damps the rank curve.
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60

    # Chroma (local, embedded).
    chroma_path: str = "data/index"

    # Pinecone (serverless).
    pinecone_api_key: str | None = None
    pinecone_index: str = "hybrid-rag"

    # Milvus / Zilliz Cloud.
    milvus_uri: str | None = None
    milvus_token: str = ""


settings = Settings()
