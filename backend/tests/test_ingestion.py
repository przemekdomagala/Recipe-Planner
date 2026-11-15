"""Tests for importing external recipes into the knowledge base."""

from app.models import Ingredient, Recipe, RecipeChunk
from app.schemas import ExternalIngredient, ExternalRecipe
from app.services.indexing import reindex_recipes
from app.services.ingestion import ingest_recipes
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeEmbedder


def make_recipe(external_id: str = "1", **overrides: object) -> ExternalRecipe:
    """Build an imported recipe with sensible defaults."""
    data: dict[str, object] = {
        "source": "themealdb",
        "external_id": external_id,
        "title": "Teriyaki Chicken",
        "instructions": "Bake the chicken.",
        "category": "Chicken",
        "area": "Japanese",
        "tags": ["Meat"],
        "ingredients": [
            ExternalIngredient(name="Soy Sauce", quantity=0.75, unit="cup"),
            ExternalIngredient(name="chicken breasts", quantity=2),
            ExternalIngredient(name="soy sauce", quantity=1, unit="tbsp"),  # duplicate
        ],
    }
    data.update(overrides)
    return ExternalRecipe.model_validate(data)


async def _count(db: AsyncSession, model: type[object]) -> int:
    """Count rows of `model`."""
    return (await db.scalar(select(func.count()).select_from(model))) or 0


async def test_ingest_creates_recipes_ingredients_and_chunks(db_session: AsyncSession) -> None:
    """Recipes are inserted with normalized, de-duplicated ingredients and get indexed."""
    report = await ingest_recipes(db_session, [make_recipe()], FakeEmbedder())

    assert (report.created, report.updated, report.indexed) == (1, 0, 1)
    names = (await db_session.scalars(select(Ingredient.name).order_by(Ingredient.name))).all()
    assert names == ["chicken breasts", "soy sauce"]
    recipe = (await db_session.scalars(select(Recipe))).one()
    assert (recipe.source, recipe.external_id, recipe.category) == ("themealdb", "1", "Chicken")
    assert await _count(db_session, RecipeChunk) == 2


async def test_reingest_is_idempotent_and_skips_unchanged(db_session: AsyncSession) -> None:
    """Running the same import twice changes nothing and embeds nothing new."""
    embedder = FakeEmbedder()
    await ingest_recipes(db_session, [make_recipe()], embedder)
    calls = embedder.calls

    report = await ingest_recipes(db_session, [make_recipe()], embedder)

    assert (report.created, report.updated, report.indexed) == (0, 1, 0)
    assert embedder.calls == calls
    assert await _count(db_session, Recipe) == 1
    assert await _count(db_session, RecipeChunk) == 2


async def test_changed_recipe_is_updated_and_reindexed(db_session: AsyncSession) -> None:
    """New text or ingredients update the recipe in place and rebuild its chunks."""
    await ingest_recipes(db_session, [make_recipe()], FakeEmbedder())

    changed = make_recipe(
        instructions="Grill the chicken.",
        ingredients=[ExternalIngredient(name="chicken breasts", quantity=3)],
    )
    report = await ingest_recipes(db_session, [changed], FakeEmbedder())

    assert report.indexed == 1
    chunks = (await db_session.scalars(select(RecipeChunk.content))).all()
    assert any("Grill the chicken." in content for content in chunks)
    assert any("3 chicken breasts" in content for content in chunks)
    assert await _count(db_session, RecipeChunk) == 2


async def test_reindex_after_embedding_model_change(db_session: AsyncSession) -> None:
    """Chunks embedded by another model are stale and get re-embedded."""
    await ingest_recipes(db_session, [make_recipe("1"), make_recipe("2")], FakeEmbedder("old"))

    stale = await reindex_recipes(db_session, FakeEmbedder("new"))
    fresh = await reindex_recipes(db_session, FakeEmbedder("new"))

    assert (stale.indexed, stale.unchanged) == (2, 0)
    assert (fresh.indexed, fresh.unchanged) == (0, 2)
    models = (await db_session.scalars(select(RecipeChunk.embedding_model).distinct())).all()
    assert models == ["new"]
