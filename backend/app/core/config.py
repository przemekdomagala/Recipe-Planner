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
    # Dimensionality of the recipe embedding vectors (1536 matches OpenAI text-embedding-3-small).
    embedding_dim: int = 1536
    sql_echo: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        """`cors_origins` split into a list, ignoring blank entries."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached `Settings` instance."""
    return Settings()


settings = get_settings()
