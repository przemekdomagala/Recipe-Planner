"""Schemas for semantic recipe search (the retrieval step of the RAG pipeline)."""

from pydantic import BaseModel, Field, field_validator


class RecipeSearchFilters(BaseModel):
    """Optional exact-match (case-insensitive) metadata filters applied before ranking."""

    category: str | None = Field(default=None, max_length=64, examples=["Seafood"])
    area: str | None = Field(default=None, max_length=64, examples=["Italian"])

    @field_validator("category", "area")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """Treat blank filters as unset."""
        return (value or "").strip() or None


class RecipeSearchRequest(BaseModel):
    """Natural-language recipe search request."""

    query: str = Field(min_length=1, max_length=500, description="Natural-language search query")
    limit: int = Field(default=5, ge=1, le=20)
    filters: RecipeSearchFilters = Field(default_factory=RecipeSearchFilters)
    min_score: float | None = Field(
        default=None, ge=-1, le=1, description="Drop hits whose cosine similarity is lower"
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        """Strip whitespace and reject blank queries."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class MatchedChunk(BaseModel):
    """A chunk of a recipe that matched the query."""

    kind: str = Field(description='"overview" or "instructions"')
    content: str
    score: float = Field(description="Cosine similarity to the query (higher is better)")


class RecipeSearchHit(BaseModel):
    """A recipe matched by the search, scored by its best-matching chunk."""

    recipe_id: int
    title: str
    category: str | None
    area: str | None
    thumbnail_url: str | None
    score: float = Field(description="Cosine similarity of the best chunk (higher is better)")
    matches: list[MatchedChunk]


class RecipeSearchResponse(BaseModel):
    """Search results for a query, best first."""

    query: str
    results: list[RecipeSearchHit]
