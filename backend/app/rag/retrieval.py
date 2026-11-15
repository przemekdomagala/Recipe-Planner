"""Semantic recipe retrieval: nearest-neighbour search over the chunk embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from app.core.config import settings
from app.llm import Embedder
from app.models import Recipe, RecipeChunk
from app.rag.chunking import ChunkKind

_HNSW_DEFAULT_EF_SEARCH = 40


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk that matched the query; `score` is the cosine similarity (higher is better)."""

    chunk_index: int
    kind: ChunkKind
    content: str
    score: float


@dataclass(frozen=True, slots=True)
class RecipeHit:
    """A recipe matched by the query, scored by its best-matching chunk."""

    recipe_id: int
    title: str
    category: str | None
    area: str | None
    image_url: str | None
    score: float
    # Matched chunks of this recipe, best first.
    chunks: tuple[RetrievedChunk, ...]


async def retrieve_recipes(
    db: AsyncSession,
    embedder: Embedder,
    query: str,
    *,
    limit: int,
    category: str | None = None,
    area: str | None = None,
    candidate_multiplier: int = settings.rag_candidate_multiplier,
) -> list[RecipeHit]:
    """Return the recipes most similar to `query`, best first.

    The `limit * candidate_multiplier` nearest chunks are fetched and grouped by recipe,
    so a recipe with many matching chunks counts once. Requires PostgreSQL + pgvector.

    Args:
        db: Database session.
        embedder: Embedding model; must be the one the chunks were indexed with.
        query: Natural-language query.
        limit: Maximum number of recipes to return.
        category: Only return recipes of this category (case-insensitive).
        area: Only return recipes of this cuisine/area (case-insensitive).
        candidate_multiplier: How many chunks to fetch per requested recipe.

    Raises:
        LLMServiceError: If the query cannot be embedded.
    """
    query_vector = await embedder.embed_query(query)
    candidates = limit * candidate_multiplier
    # An HNSW index scan yields at most `hnsw.ef_search` rows (default 40), however large
    # the LIMIT; raise it for this transaction so every requested candidate can be returned.
    await db.execute(
        select(
            func.set_config("hnsw.ef_search", str(max(_HNSW_DEFAULT_EF_SEARCH, candidates)), True)
        )
    )
    distance = RecipeChunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(RecipeChunk, distance)
        .join(RecipeChunk.recipe)
        .options(contains_eager(RecipeChunk.recipe))
        .where(RecipeChunk.embedding_model == embedder.model_name)
        .order_by(distance)
        .limit(candidates)
    )
    if category:
        stmt = stmt.where(func.lower(Recipe.category) == category.lower())
    if area:
        stmt = stmt.where(func.lower(Recipe.area) == area.lower())

    grouped: dict[int, list[tuple[RecipeChunk, float]]] = {}
    for chunk, chunk_distance in await db.execute(stmt):
        grouped.setdefault(chunk.recipe_id, []).append((chunk, float(chunk_distance)))

    hits: list[RecipeHit] = []
    # Dicts keep insertion order, and rows arrive nearest first, so recipes are ranked.
    for matches in list(grouped.values())[:limit]:
        recipe = matches[0][0].recipe
        retrieved = tuple(
            RetrievedChunk(
                chunk_index=chunk.chunk_index,
                kind=ChunkKind(chunk.kind),
                content=chunk.content,
                score=1.0 - chunk_distance,
            )
            for chunk, chunk_distance in matches
        )
        hits.append(
            RecipeHit(
                recipe_id=recipe.id,
                title=recipe.title,
                category=recipe.category,
                area=recipe.area,
                image_url=recipe.image_url,
                score=retrieved[0].score,
                chunks=retrieved,
            )
        )
    return hits


async def fetch_summary_chunks(db: AsyncSession, recipe_ids: Sequence[int]) -> dict[int, str]:
    """Return the summary chunk text of each given recipe that has one, keyed by recipe id."""
    if not recipe_ids:
        return {}
    rows = await db.execute(
        select(RecipeChunk.recipe_id, RecipeChunk.content).where(
            RecipeChunk.recipe_id.in_(recipe_ids), RecipeChunk.kind == ChunkKind.SUMMARY.value
        )
    )
    return dict(rows.all())
