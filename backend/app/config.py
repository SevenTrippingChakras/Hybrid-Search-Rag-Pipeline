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

    # Contextual Retrieval (Anthropic): before embedding, prepend an LLM-generated
    # blurb that situates each chunk in its document. Off by default -- it costs
    # one LLM call per chunk at ingest; turn on to evaluate the retrieval lift.
    contextual_retrieval: bool = False

    # Verification: judge each cited claim concurrently, capped so a burst does
    # not trip OpenAI rate limits. 1 = sequential (old behaviour).
    verification_max_workers: int = 8

    # Confidence: weights fusing the three answer-trust signals (retrieval
    # relevance, citation coverage, question completeness) into one composite.
    # They should sum to 1.0.
    confidence_retrieval_weight: float = 0.4
    confidence_citation_weight: float = 0.4
    confidence_completeness_weight: float = 0.2

    # Abstention: if retrieval confidence (top rerank score, 0-1) is below this,
    # the pipeline returns a structured "I don't know" instead of generating.
    abstain_threshold: float = 0.2

    # Object storage for uploaded documents. Only the S3-compatible adapter for
    # now; the factory stays so a future backend remains swappable. Values point
    # at Supabase Storage today (S3-compatible); swapping to R2 or AWS S3 later is
    # a credentials change, not a code change.
    storage_backend: str = "s3"
    s3_endpoint: str | None = None
    s3_region: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_bucket: str | None = None

    # Metadata store for document records (status, storage_key, chunk_count).
    # Local dev is the Mongo container in docker-compose.yml; prod is a managed
    # MongoDB. Embeddings never live here - those stay in OpenSearch.
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "hybrid_rag"

    # Browser origins allowed to call the API (CORS). Comma-separated. The Vite
    # dev server runs on 5173 by default.
    cors_origins: str = "http://localhost:5173"


settings = Settings()
