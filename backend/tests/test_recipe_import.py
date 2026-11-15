"""Tests for importing external recipes into the database."""

from collections.abc import AsyncIterator, Iterable
from dataclasses import replace
from typing import Any

from app.models import Ingredient, RecipeChunk
from app.services import import_recipe, import_recipes
from app.services.recipe_import import IngredientLine, build_ingredient_lines
from app.sources import ExternalIngredient, ExternalRecipe
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeEmbedder


def make_recipe(external_id: str = "1", **overrides: Any) -> ExternalRecipe:
    """Build an external recipe with two ingredients; `overrides` replace fields."""
    recipe = ExternalRecipe(
        source="themealdb",
        external_id=external_id,
        title="Garlic Pasta",
        instructions="Boil pasta. Fry garlic. Combine.",
        category="Pasta",
        area="Italian",
        tags=("Quick",),
        ingredients=(
            ExternalIngredient("Spaghetti", "200g"),
            ExternalIngredient("Garlic", "3 cloves minced"),
        ),
    )
    return replace(recipe, **overrides)


async def _aiter(items: Iterable[ExternalRecipe]) -> AsyncIterator[ExternalRecipe]:
    """Turn an iterable into an async iterator, like a source client yields recipes."""
    for item in items:
        yield item


def test_build_ingredient_lines_normalizes_and_parses() -> None:
    """Names are normalized and measures parsed, keeping the raw measure."""
    lines = build_ingredient_lines([ExternalIngredient("  Olive  Oil ", "2 tblsp")])

    assert lines == [IngredientLine("olive oil", 2.0, "tbsp", "2 tblsp")]


def test_build_ingredient_lines_merges_duplicates() -> None:
    """Repeated ingredients merge: same units are summed, different units become ambiguous."""
    lines = build_ingredient_lines(
        [
            ExternalIngredient("Salt", "1 tsp"),
            ExternalIngredient("Sugar", "100g"),
            ExternalIngredient("salt", "2 tsp"),
            ExternalIngredient("Sugar", "1 cup"),
            ExternalIngredient("Sugar", None),
        ]
    )

    assert lines == [
        IngredientLine("salt", 3.0, "tsp", "1 tsp + 2 tsp"),
        IngredientLine("sugar", None, None, "100g + 1 cup"),
    ]


async def test_import_recipe_creates_recipe_and_ingredients(db_session: AsyncSession) -> None:
    """A new recipe is stored with metadata and ingredient links in source order."""
    recipe, created = await import_recipe(db_session, make_recipe())

    assert created is True
    assert (recipe.source, recipe.external_id) == ("themealdb", "1")
    assert (recipe.category, recipe.area, recipe.tags) == ("Pasta", "Italian", ["Quick"])
    assert [
        (link.ingredient.name, link.quantity, link.unit, link.measure)
        for link in recipe.ingredients
    ] == [("spaghetti", 200.0, "g", "200g"), ("garlic", 3.0, "clove", "3 cloves minced")]


async def test_reimport_updates_instead_of_duplicating(db_session: AsyncSession) -> None:
    """Re-importing the same external id updates the recipe and its ingredient links."""
    first, _ = await import_recipe(db_session, make_recipe())

    updated, created = await import_recipe(
        db_session,
        make_recipe(
            title="Garlic Pasta v2",
            ingredients=(
                ExternalIngredient("Garlic", "4 cloves"),
                ExternalIngredient("Chilli", "1"),
            ),
        ),
    )

    assert created is False
    assert updated.id == first.id
    assert updated.title == "Garlic Pasta v2"
    assert [(link.ingredient.name, link.quantity) for link in updated.ingredients] == [
        ("garlic", 4.0),
        ("chilli", 1.0),
    ]


async def test_import_reuses_existing_ingredients(db_session: AsyncSession) -> None:
    """Ingredients shared by recipes are stored once."""
    await import_recipe(db_session, make_recipe("1"))
    await import_recipe(db_session, make_recipe("2", title="Other"))

    count = await db_session.scalar(select(func.count()).select_from(Ingredient))
    assert count == 2


async def test_import_recipes_indexes_and_respects_limit(db_session: AsyncSession) -> None:
    """Imported recipes are indexed right away; `limit` caps the number imported."""
    embedder = FakeEmbedder()

    result = await import_recipes(
        db_session,
        _aiter([make_recipe("1"), make_recipe("2"), make_recipe("3")]),
        embedder=embedder,
        limit=2,
    )

    assert (result.created, result.updated, result.indexed) == (2, 0, 2)
    recipe_ids = await db_session.scalars(select(func.distinct(RecipeChunk.recipe_id)))
    assert len(recipe_ids.all()) == 2


async def test_import_recipes_without_embedder_skips_indexing(db_session: AsyncSession) -> None:
    """With no embedder, recipes are imported but not indexed."""
    result = await import_recipes(db_session, _aiter([make_recipe()]))

    assert (result.created, result.indexed) == (1, 0)
    assert await db_session.scalar(select(func.count()).select_from(RecipeChunk)) == 0
