"""Schemas for the (placeholder) RAG recipe search."""

from pydantic import BaseModel, Field


class RecipeSearchRequest(BaseModel):
    """Natural-language recipe search request."""

    query: str = Field(min_length=1, max_length=500, description="Natural-language search query")
    limit: int = Field(default=5, ge=1, le=50)


class RecipeSearchHit(BaseModel):
    """A single recipe matched by the search."""

    recipe_id: int
    title: str
    score: float = Field(description="Similarity score (higher is better)")


class RecipeSearchResponse(BaseModel):
    """Search results for a query."""

    query: str
    results: list[RecipeSearchHit]
    mock: bool = Field(description="True while the response is a placeholder, not real RAG output")
