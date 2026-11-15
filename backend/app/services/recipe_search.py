"""Recipe search: the retrieval step of the RAG pipeline.

The query is embedded, the nearest chunks are fetched from the vector store and grouped
by recipe; a recipe's score is the similarity of its best-matching chunk.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.crud.chunk import ScoredChunk
from app.schemas import MatchedChunk, RecipeSearchHit, RecipeSearchRequest, RecipeSearchResponse
from app.services.llm import Embedder


async def search_recipes(
    db: AsyncSession, request: RecipeSearchRequest, embedder: Embedder
) -> RecipeSearchResponse:
    """Return the recipes most relevant to `request.query`, best first.

    Raises:
        LLMError: If the query can't be embedded.
    """
    query_vector = await embedder.embed_query(request.query)
    scored = await crud.chunk.nearest_chunks(
        db,
        query_vector,
        k=request.limit * settings.search_candidates_per_hit,
        filters=request.filters,
    )
    hits = group_by_recipe(scored)
    if request.min_score is not None:
        hits = [hit for hit in hits if hit.score >= request.min_score]
    return RecipeSearchResponse(query=request.query, results=hits[: request.limit])


def group_by_recipe(scored: list[ScoredChunk]) -> list[RecipeSearchHit]:
    """Merge chunk hits into one hit per recipe, ordered by the best chunk's score.

    `scored` must be sorted best first, as returned by `crud.chunk.nearest_chunks`.
    """
    hits: dict[int, RecipeSearchHit] = {}
    for item in scored:
        recipe = item.chunk.recipe
        match = MatchedChunk(kind=item.chunk.kind, content=item.chunk.content, score=item.score)
        if recipe.id in hits:
            hits[recipe.id].matches.append(match)
            continue
        hits[recipe.id] = RecipeSearchHit(
            recipe_id=recipe.id,
            title=recipe.title,
            category=recipe.category,
            area=recipe.area,
            thumbnail_url=recipe.thumbnail_url,
            score=item.score,
            matches=[match],
        )
    return list(hits.values())
