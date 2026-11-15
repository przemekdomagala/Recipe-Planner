"""Schemas for the recipe assistant (retrieval-augmented LLM answers)."""

from pydantic import BaseModel, Field, field_validator

from app.schemas.search import RecipeSearchFilters, RecipeSearchHit


class AskRequest(BaseModel):
    """A cooking question answered from the recipe knowledge base."""

    question: str = Field(
        min_length=1, max_length=1000, examples=["What can I cook with chicken and rice?"]
    )
    max_recipes: int = Field(
        default=3, ge=1, le=8, description="How many retrieved recipes the LLM gets as context"
    )
    filters: RecipeSearchFilters = Field(default_factory=RecipeSearchFilters)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        """Strip whitespace and reject blank questions."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class AskResponse(BaseModel):
    """The LLM's answer plus the recipes it was grounded on."""

    question: str
    answer: str
    model: str | None = Field(description="Chat model that wrote the answer; null if none ran")
    sources: list[RecipeSearchHit]
