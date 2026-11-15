"""Schemas for semantic recipe search."""

from pydantic import BaseModel, Field


class RecipeSearchRequest(BaseModel):
    """Natural-language recipe search request with optional metadata filters."""

    query: str = Field(min_length=1, max_length=500, description="Natural-language search query")
    limit: int = Field(default=5, ge=1, le=20)
    category: str | None = Field(default=None, max_length=64, examples=["Vegetarian"])
    area: str | None = Field(default=None, max_length=64, examples=["Italian"])


class RecipeSearchHit(BaseModel):
    """A recipe matched by the search."""

    recipe_id: int
    title: str
    category: str | None
    area: str | None
    image_url: str | None
    score: float = Field(
        description="Cosine similarity of the best-matching chunk (higher is better)"
    )
    snippet: str = Field(description="Beginning of the best-matching chunk")


class RecipeSearchResponse(BaseModel):
    """Search results for a query, best first."""

    query: str
    results: list[RecipeSearchHit]
