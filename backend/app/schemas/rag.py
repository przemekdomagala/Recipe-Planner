"""Schemas for the RAG assistant endpoints."""

from pydantic import BaseModel, Field

from app.schemas.search import RecipeSearchHit


class AskRequest(BaseModel):
    """A natural-language question answered from the recipe knowledge base."""

    question: str = Field(
        min_length=1, max_length=1000, examples=["What can I cook with chicken and rice?"]
    )
    top_k: int = Field(default=4, ge=1, le=10, description="Recipes passed to the model as context")
    category: str | None = Field(default=None, max_length=64)
    area: str | None = Field(default=None, max_length=64)


class AskResponse(BaseModel):
    """The generated answer and the recipes it is grounded on."""

    question: str
    answer: str
    model: str | None = Field(description="Chat model used; null if no recipe matched")
    sources: list[RecipeSearchHit]


class RagStatus(BaseModel):
    """Readiness of the knowledge base and the LLM service."""

    total_recipes: int
    indexed_recipes: int
    chunks: int
    embedding_model: str
    chat_model: str
    llm_available: bool = Field(description="Ollama is reachable and both models are pulled")
    missing_models: list[str]
    detail: str | None = Field(default=None, description="Why the LLM is unavailable, if it is")
