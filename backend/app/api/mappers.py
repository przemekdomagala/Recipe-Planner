"""Conversions from service-layer results to API response schemas."""

from app.rag import RecipeHit
from app.schemas import RecipeSearchHit

SNIPPET_MAX_CHARS = 300


def to_search_hit(hit: RecipeHit) -> RecipeSearchHit:
    """Build the API representation of a retrieval hit, with a snippet of its best chunk.

    Args:
        hit: Recipe returned by the retrieval pipeline.
    """
    best = hit.chunks[0].content if hit.chunks else ""
    snippet = best if len(best) <= SNIPPET_MAX_CHARS else best[: SNIPPET_MAX_CHARS - 1] + "…"
    return RecipeSearchHit(
        recipe_id=hit.recipe_id,
        title=hit.title,
        category=hit.category,
        area=hit.area,
        image_url=hit.image_url,
        score=round(hit.score, 4),
        snippet=snippet,
    )
