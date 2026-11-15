"""Building the vector store: chunking recipes, embedding the chunks and storing them."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.llm import Embedder
from app.models import Recipe, RecipeChunk
from app.rag.chunking import RecipeDocument, chunk_recipe


@dataclass(frozen=True, slots=True)
class IndexStats:
    """How much of the recipe collection is searchable."""

    total_recipes: int
    indexed_recipes: int
    chunks: int


@dataclass(slots=True)
class ReindexResult:
    """Outcome of `reindex_recipes`."""

    indexed: int = 0
    unchanged: int = 0


async def index_recipe(
    db: AsyncSession,
    recipe: Recipe,
    embedder: Embedder,
    *,
    force: bool = False,
    max_chars: int = settings.chunk_max_chars,
    overlap_chars: int = settings.chunk_overlap_chars,
) -> bool:
    """Replace a recipe's chunks with freshly embedded ones and commit.

    Embedding is skipped when the stored chunks already have exactly the new texts and
    were produced by the same embedding model, which makes re-imports cheap. On failure
    the previous chunks are kept.

    Args:
        db: Database session.
        recipe: Recipe with its ingredients eagerly loaded.
        embedder: Embedding model.
        force: Re-embed even if the stored chunks are up to date.
        max_chars: Maximum instruction text per chunk.
        overlap_chars: Overlap between consecutive instruction chunks.

    Returns:
        `True` if the chunks were (re)written, `False` if they were already up to date.

    Raises:
        LLMServiceError: If embedding fails.
    """
    chunks = chunk_recipe(
        RecipeDocument.from_recipe(recipe), max_chars=max_chars, overlap_chars=overlap_chars
    )
    if not force:
        stored = await db.execute(
            select(RecipeChunk.content, RecipeChunk.embedding_model)
            .where(RecipeChunk.recipe_id == recipe.id)
            .order_by(RecipeChunk.chunk_index)
        )
        wanted = [(chunk.content, embedder.model_name) for chunk in chunks]
        if [tuple(row) for row in stored.all()] == wanted:
            return False

    vectors = await embedder.embed_documents([chunk.content for chunk in chunks])
    await db.execute(delete(RecipeChunk).where(RecipeChunk.recipe_id == recipe.id))
    db.add_all(
        RecipeChunk(
            recipe_id=recipe.id,
            chunk_index=chunk.index,
            kind=chunk.kind.value,
            content=chunk.content,
            embedding_model=embedder.model_name,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    )
    await db.commit()
    return True


async def reindex_recipes(
    db: AsyncSession,
    embedder: Embedder,
    *,
    force: bool = False,
    batch_size: int = 100,
    on_progress: Callable[[int], None] | None = None,
) -> ReindexResult:
    """(Re)index every recipe, skipping those whose chunks are up to date unless `force`.

    Args:
        db: Database session.
        embedder: Embedding model.
        force: Re-embed every recipe even if its chunks are up to date. Not needed after
            changing the embedding model or chunking settings; both are detected.
        batch_size: Number of recipes loaded from the database at a time.
        on_progress: Called with the number of recipes processed so far after each one.

    Raises:
        LLMServiceError: If embedding fails; recipes processed before stay indexed.
    """
    result = ReindexResult()
    offset = 0
    while recipes := await crud.recipe.get_recipes(db, skip=offset, limit=batch_size):
        for recipe in recipes:
            if await index_recipe(db, recipe, embedder, force=force):
                result.indexed += 1
            else:
                result.unchanged += 1
            if on_progress is not None:
                on_progress(result.indexed + result.unchanged)
        offset += batch_size
    return result


async def get_index_stats(db: AsyncSession) -> IndexStats:
    """Count all recipes, the recipes that have chunks, and the chunks themselves."""
    total_recipes = await db.scalar(select(func.count()).select_from(Recipe))
    indexed_recipes = await db.scalar(select(func.count(func.distinct(RecipeChunk.recipe_id))))
    chunks = await db.scalar(select(func.count()).select_from(RecipeChunk))
    return IndexStats(
        total_recipes=total_recipes or 0, indexed_recipes=indexed_recipes or 0, chunks=chunks or 0
    )
