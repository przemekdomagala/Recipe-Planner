"""Application settings loaded from environment variables or a `.env` file."""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables (or a local `.env`)."""

    # The repo-root `.env` (shared with Docker Compose) is read when running from `backend/`;
    # a `backend/.env` overrides it, and real environment variables override both.
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Recipe Planner API"
    database_url: str = "postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_planner"
    # Comma-separated list of allowed CORS origins.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    sql_echo: bool = False

    # --- Ollama (LLM + embeddings) ------------------------------------------------------
    # The local Ollama app. "-cloud" models (e.g. gemma4:31b-cloud) are run on Ollama Cloud
    # by the app itself, using the account it's signed in with (`ollama signin`).
    ollama_base_url: str = "http://localhost:11434"
    # Only for calling https://ollama.com directly (sent as `Authorization: Bearer <key>`).
    ollama_api_key: SecretStr | None = None
    # Embeddings can use a different Ollama server; unset means the chat server.
    embedding_base_url: str | None = None
    embedding_api_key: SecretStr | None = None
    ollama_timeout_seconds: float = Field(default=300.0, gt=0)
    # Chat model used for answer generation.
    llm_model: str = "qwen3.5:4b"
    # Reasoning ("thinking") mode; off by default because on CPU it multiplies answer time.
    llm_think: bool = False
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    # Hard cap on answer length in tokens (Ollama `num_predict`); small models ramble, and on
    # CPU every token costs time.
    llm_max_tokens: int = Field(default=350, gt=0)
    # Embedding model; `embedding_dim` must match its output size and be set before the
    # `recipe_chunks` table is first created (768 = nomic-embed-text).
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = Field(default=768, gt=0)
    # Task prefixes some embedding models are trained with (nomic-embed-text uses these).
    # Set both to "" for models that don't expect them.
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    embedding_batch_size: int = Field(default=32, gt=0)

    # --- RAG ----------------------------------------------------------------------------
    # Upper bound for one instructions chunk, in characters.
    chunk_max_chars: int = Field(default=1000, ge=200)
    # Chunks fetched per requested recipe before grouping by recipe.
    search_candidates_per_hit: int = Field(default=4, ge=1)

    # --- Data sources -------------------------------------------------------------------
    # "1" is TheMealDB's public test key.
    themealdb_base_url: str = "https://www.themealdb.com/api/json/v1/1"

    @field_validator("ollama_api_key", "embedding_base_url", "embedding_api_key", mode="before")
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        """Treat empty values (e.g. `OLLAMA_API_KEY=` passed through by Compose) as unset."""
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into a list, ignoring blank entries."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def embedding_server_url(self) -> str:
        """Ollama server used for embeddings (falls back to `ollama_base_url`)."""
        return self.embedding_base_url or self.ollama_base_url

    @property
    def embedding_server_key(self) -> SecretStr | None:
        """API key for the embedding server.

        Falls back to `ollama_api_key`, but only when embeddings use the same server, so a
        cloud key is never sent to some other host.
        """
        if self.embedding_api_key is not None:
            return self.embedding_api_key
        same_server = self.embedding_base_url in (None, self.ollama_base_url)
        return self.ollama_api_key if same_server else None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached `Settings` instance."""
    return Settings()


settings = get_settings()
