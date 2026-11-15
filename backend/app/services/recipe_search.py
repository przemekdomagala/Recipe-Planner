"""Recipe search service: home of the future RAG retrieval pipeline."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import RecipeSearchHit, RecipeSearchRequest, RecipeSearchResponse


async def search_recipes(db: AsyncSession, request: RecipeSearchRequest) -> RecipeSearchResponse:
    """Placeholder for the RAG retrieval step.

    TODO: embed `request.query`, run a pgvector nearest-neighbour query against
    `Recipe.embedding` (e.g. `Recipe.embedding.cosine_distance(query_vector)`), and pass the
    retrieved recipes to the LLM. Until then a fixed mock payload is returned.
    """
    del db  # Unused until the vector search is implemented.
    mock_hits = [
        RecipeSearchHit(recipe_id=0, title="Mock recipe (vector search not implemented)", score=0.0)
    ]
    return RecipeSearchResponse(query=request.query, results=mock_hits[: request.limit], mock=True)
