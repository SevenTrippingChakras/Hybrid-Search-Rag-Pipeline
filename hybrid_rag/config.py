"""Central configuration for the pipeline.

One ``Settings`` class reads every knob from the environment (and ``.env``).
``settings`` is a shared instance; import it where you need a value. Optional
fields default to ``None`` so the module imports without a backend's credentials.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None

    # Which retrieval backend the store uses. Only opensearch for now; the factory
    # stays so a future adapter (e.g. Azure AI Search) remains swappable.
    vector_backend: str = "opensearch"

    # OpenSearch: dense k-NN + sparse BM25 in one engine. Local dev is the Docker
    # single node from docker-compose.yml; prod is Amazon OpenSearch.
    opensearch_host: str = "http://localhost:9200"
    opensearch_index: str = "chunks"

    # Deduplication: skip a chunk whose cosine similarity to an existing chunk
    # exceeds this. Set >= 1.0 to disable dedup entirely.
    dedup_threshold: float = 0.95

    # Hybrid fusion: OpenSearch's native RRF (score-ranker-processor) rank
    # constant. Larger values flatten the rank curve; smaller sharpen it.
    rrf_k: int = 60

    # Reranker: cross-encoder that re-scores fused candidates, keeping the
    # best rerank_top_k.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 5

    # Generation: which LLM provider and model produce the grounded answer.
    # llm_backend is the seam for a future adapter (Anthropic, Bedrock);
    # only openai for now, mirroring vector_backend.
    llm_backend: str = "openai"
    generation_model: str = "gpt-4o-mini"


settings = Settings()
