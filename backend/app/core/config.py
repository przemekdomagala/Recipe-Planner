"""Application settings loaded from environment variables or a `.env` file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables (or a local `.env`)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Recipe Planner API"
    database_url: str = "postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_planner"
    # Comma-separated list of allowed CORS origins.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    sql_echo: bool = False

    # --- LLM (Ollama) ---------------------------------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    # Generous because the first request after start-up also loads the model into memory.
    ollama_timeout_seconds: float = 180.0
    # Context window requested from Ollama; must fit the system prompt, the retrieved
    # recipe chunks and the answer.
    ollama_num_ctx: int = 8192
    llm_temperature: float = 0.2

    # --- Embeddings -----------------------------------------------------------------------
    # Must match the output size of `ollama_embedding_model` (768 for nomic-embed-text) and be
    # set before the `recipe_chunks` table is first created.
    embedding_dim: int = 768
    # nomic-embed-text is trained with task prefixes; set both to "" for models without them.
    embedding_document_prefix: str = "search_document: "
    embedding_query_prefix: str = "search_query: "
    embedding_batch_size: int = 32

    # --- RAG ------------------------------------------------------------------------------
    chunk_max_chars: int = 1000
    chunk_overlap_chars: int = 150
    # Retrieval fetches `limit * rag_candidate_multiplier` nearest chunks, then keeps the best
    # chunk per recipe, so several chunks of one recipe can't crowd out other recipes.
    rag_candidate_multiplier: int = 5

    # --- Recipe sources -------------------------------------------------------------------
    # "1" is TheMealDB's public test key; a supporter key can be put in the URL instead.
    themealdb_base_url: str = "https://www.themealdb.com/api/json/v1/1"
    themealdb_timeout_seconds: float = 30.0

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into a list, ignoring blank entries."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached `Settings` instance."""
    return Settings()


settings = get_settings()
