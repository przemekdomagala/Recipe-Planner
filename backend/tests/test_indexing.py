"""Tests for building the chunk vector store."""

import pytest
from app import crud
from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.models import Recipe, RecipeChunk
from app.rag import get_index_stats, index_recipe, reindex_recipes
from app.schemas import RecipeCreate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeEmbedder


async def _create_recipe(db: AsyncSession, title: str = "Soup") -> Recipe:
    """Create a recipe without ingredients."""
    return await crud.recipe.create_recipe(
        db, RecipeCreate(title=title, instructions="Chop vegetables. Simmer for an hour.")
    )


async def _chunks(db: AsyncSession, recipe_id: int) -> list[RecipeChunk]:
    """Return the stored chunks of a recipe in order."""
    result = await db.scalars(
        select(RecipeChunk)
        .where(RecipeChunk.recipe_id == recipe_id)
        .order_by(RecipeChunk.chunk_index)
    )
    return list(result.all())


async def test_index_recipe_stores_chunks_with_embeddings(db_session: AsyncSession) -> None:
    """Chunks are stored with their kind, model name and full-size vector."""
    recipe = await _create_recipe(db_session)

    assert await index_recipe(db_session, recipe, FakeEmbedder()) is True

    chunks = await _chunks(db_session, recipe.id)
    assert [chunk.kind for chunk in chunks] == ["summary", "instructions"]
    assert all(chunk.embedding_model == "fake-embedder" for chunk in chunks)
    assert all(len(chunk.embedding) == settings.embedding_dim for chunk in chunks)


async def test_index_recipe_skips_unchanged_recipes(db_session: AsyncSession) -> None:
    """Re-indexing an unchanged recipe does not call the embedder, unless forced."""
    recipe = await _create_recipe(db_session)
    embedder = FakeEmbedder()
    await index_recipe(db_session, recipe, embedder)
    calls = embedder.calls

    assert await index_recipe(db_session, recipe, embedder) is False
    assert embedder.calls == calls
    assert await index_recipe(db_session, recipe, embedder, force=True) is True
    assert embedder.calls == calls + 1


async def test_index_recipe_reembeds_after_a_change(db_session: AsyncSession) -> None:
    """Changing the text replaces the chunks instead of adding to them."""
    recipe = await _create_recipe(db_session)
    embedder = FakeEmbedder()
    await index_recipe(db_session, recipe, embedder)

    recipe.instructions = "Just heat it up."
    await db_session.commit()

    assert await index_recipe(db_session, recipe, embedder) is True
    chunks = await _chunks(db_session, recipe.id)
    assert len(chunks) == 2
    assert chunks[1].content.endswith("Just heat it up.")


async def test_index_recipe_keeps_old_chunks_when_embedding_fails(
    db_session: AsyncSession,
) -> None:
    """A failed embedding leaves the previous chunks in place."""
    recipe = await _create_recipe(db_session)
    embedder = FakeEmbedder()
    await index_recipe(db_session, recipe, embedder)
    embedder.fail = True

    with pytest.raises(LLMServiceError):
        await index_recipe(db_session, recipe, embedder, force=True)

    assert len(await _chunks(db_session, recipe.id)) == 2


async def test_reindex_recipes_and_stats(db_session: AsyncSession) -> None:
    """Reindexing covers all recipes once; stats reflect the vector store."""
    for title in ("A", "B", "C"):
        await _create_recipe(db_session, title)
    embedder = FakeEmbedder()

    first = await reindex_recipes(db_session, embedder, batch_size=2)
    second = await reindex_recipes(db_session, embedder, batch_size=2)

    assert (first.indexed, first.unchanged) == (3, 0)
    assert (second.indexed, second.unchanged) == (0, 3)
    stats = await get_index_stats(db_session)
    assert (stats.total_recipes, stats.indexed_recipes, stats.chunks) == (3, 3, 6)
