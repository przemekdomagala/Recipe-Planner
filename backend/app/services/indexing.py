"""Indexing: chunk a recipe, embed the chunks and store them in the vector store."""

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.core.exceptions import LLMError
from app.models import Recipe, RecipeChunk
from app.services.chunking import RecipeDocument, chunk_recipe
from app.services.llm import Embedder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReindexReport:
    """Outcome of a bulk (re)indexing run."""

    indexed: int = 0
    unchanged: int = 0


def is_up_to_date(recipe: Recipe, embedding_model: str) -> bool:
    """Whether the stored chunks match the recipe's current text and embedding model.

    `recipe` needs its ingredients and chunks loaded.
    """
    drafts = chunk_recipe(RecipeDocument.from_recipe(recipe), max_chars=settings.chunk_max_chars)
    stored = [(chunk.kind, chunk.content, chunk.embedding_model) for chunk in recipe.chunks]
    return stored == [(draft.kind, draft.content, embedding_model) for draft in drafts]


async def index_recipe(
    db: AsyncSession, recipe: Recipe, embedder: Embedder, *, force: bool = False
) -> bool:
    """(Re)build a recipe's chunks and embeddings, skipping the work if nothing changed.

    Args:
        db: Database session.
        recipe: Recipe with ingredients and chunks loaded (see `crud.recipe.get_recipe`).
        embedder: Embedding model.
        force: Re-embed even if the stored chunks are up to date.

    Returns:
        `True` if the recipe was (re)indexed, `False` if it was already up to date.

    Raises:
        LLMError: If embedding fails; the existing chunks are left untouched.
    """
    if not force and is_up_to_date(recipe, embedder.embedding_model):
        return False

    drafts = chunk_recipe(RecipeDocument.from_recipe(recipe), max_chars=settings.chunk_max_chars)
    vectors = await embedder.embed_documents([draft.content for draft in drafts])
    recipe.chunks = [
        RecipeChunk(
            position=position,
            kind=draft.kind,
            content=draft.content,
            embedding_model=embedder.embedding_model,
            embedding=vector,
        )
        for position, (draft, vector) in enumerate(zip(drafts, vectors, strict=True))
    ]
    await db.commit()
    return True


async def try_index_recipe(db: AsyncSession, recipe_id: int, embedder: Embedder) -> bool:
    """Index a recipe, logging (not raising) LLM failures.

    Used right after a recipe is created through the API: the recipe is saved either way,
    and `python -m app.cli reindex` catches up on anything that couldn't be embedded.

    Returns:
        `True` if the recipe was indexed.
    """
    recipe = await crud.recipe.get_recipe(db, recipe_id, with_chunks=True)
    if recipe is None:
        return False
    try:
        return await index_recipe(db, recipe, embedder)
    except LLMError as exc:
        await db.rollback()
        logger.warning("Recipe %s saved but not indexed: %s", recipe_id, exc)
        return False


async def reindex_recipes(
    db: AsyncSession, embedder: Embedder, *, force: bool = False
) -> ReindexReport:
    """Index every recipe whose chunks are missing or stale (or all, with `force`).

    Each recipe is committed separately, so an interrupted run keeps its progress.
    """
    report = ReindexReport()
    for recipe_id in await crud.recipe.get_recipe_ids(db):
        recipe = await crud.recipe.get_recipe(db, recipe_id, with_chunks=True)
        if recipe is None:
            continue
        if await index_recipe(db, recipe, embedder, force=force):
            report.indexed += 1
        else:
            report.unchanged += 1
    return report
