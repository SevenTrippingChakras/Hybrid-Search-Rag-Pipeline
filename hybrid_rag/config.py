"""Central configuration for the pipeline.

One ``Settings`` class reads every knob from the environment (and ``.env``), so
config lives in a single validated place instead of scattered ``os.environ``
calls. As the project grows (the Phase 5 FastAPI service, the eval harness) new
settings — host, port, model names — land here too.

``settings`` is a shared instance; import it where you need a value. Optional
fields default to ``None`` so the module imports even when a given backend's
credentials are absent (the store fails only when that backend is actually used).
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

    # Chroma (local, embedded).
    chroma_path: str = "data/index"

    # Pinecone (serverless).
    pinecone_api_key: str | None = None
    pinecone_index: str = "hybrid-rag"

    # Milvus / Zilliz Cloud.
    milvus_uri: str | None = None
    milvus_token: str = ""


settings = Settings()
