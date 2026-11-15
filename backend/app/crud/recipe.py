"""Database access for recipes and their ingredient links."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import IngredientsNotFoundError
from app.crud.ingredient import get_or_create_ingredients
from app.models import Ingredient, Recipe, RecipeIngredient
from app.schemas import ExternalRecipe, RecipeCreate, normalize_ingredient_name

_WITH_INGREDIENTS = selectinload(Recipe.ingredients).selectinload(RecipeIngredient.ingredient)
# Chunk metadata only: `RecipeChunk.embedding` is deferred and never needed here.
_WITH_CHUNKS = selectinload(Recipe.chunks)


async def get_recipe(
    db: AsyncSession, recipe_id: int, *, with_chunks: bool = False
) -> Recipe | None:
    """Return a recipe with its ingredients eagerly loaded, or `None` if missing.

    `populate_existing` refreshes an instance already in the session's identity map, so
    relationships added in the same session are reflected.

    Args:
        db: Database session.
        recipe_id: Recipe primary key.
        with_chunks: Also load the recipe's chunks (without their embeddings).
    """
    options = [_WITH_INGREDIENTS, _WITH_CHUNKS] if with_chunks else [_WITH_INGREDIENTS]
    result = await db.scalars(
        select(Recipe)
        .where(Recipe.id == recipe_id)
        .options(*options)
        .execution_options(populate_existing=True)
    )
    return result.one_or_none()


async def get_recipes(db: AsyncSession, *, skip: int = 0, limit: int = 100) -> Sequence[Recipe]:
    """Return recipes ordered by id, with their ingredients eagerly loaded.

    Args:
        db: Database session.
        skip: Number of rows to skip.
        limit: Maximum number of rows to return.
    """
    result = await db.scalars(
        select(Recipe).order_by(Recipe.id).offset(skip).limit(limit).options(_WITH_INGREDIENTS)
    )
    return result.all()


async def get_recipes_by_ids(db: AsyncSession, recipe_ids: Sequence[int]) -> list[Recipe]:
    """Return the given recipes with ingredients loaded, in the order of `recipe_ids`.

    Unknown ids are skipped.
    """
    if not recipe_ids:
        return []
    result = await db.scalars(
        select(Recipe).where(Recipe.id.in_(recipe_ids)).options(_WITH_INGREDIENTS)
    )
    by_id = {recipe.id: recipe for recipe in result}
    return [by_id[recipe_id] for recipe_id in recipe_ids if recipe_id in by_id]


async def get_recipe_ids(db: AsyncSession) -> list[int]:
    """Return the ids of all recipes, ascending."""
    result = await db.scalars(select(Recipe.id).order_by(Recipe.id))
    return list(result)


async def create_recipe(db: AsyncSession, data: RecipeCreate) -> Recipe:
    """Insert a recipe and its ingredient links in one transaction.

    Args:
        db: Database session.
        data: Validated recipe payload.

    Returns:
        The persisted recipe, re-loaded with its ingredients.

    Raises:
        IngredientsNotFoundError: If any referenced ingredient id does not exist (nothing
            is written in that case).
    """
    requested_ids = {item.ingredient_id for item in data.ingredients}
    if requested_ids:
        found = await db.scalars(select(Ingredient.id).where(Ingredient.id.in_(requested_ids)))
        missing = sorted(requested_ids - set(found.all()))
        if missing:
            raise IngredientsNotFoundError(missing)

    recipe = Recipe(
        title=data.title,
        instructions=data.instructions,
        category=data.category,
        area=data.area,
        tags=data.tags,
        ingredients=[
            RecipeIngredient(
                ingredient_id=item.ingredient_id, quantity=item.quantity, unit=item.unit
            )
            for item in data.ingredients
        ],
    )
    db.add(recipe)
    await db.commit()

    # Re-load with eager relationships: lazy loading is not allowed in async sessions.
    created = await get_recipe(db, recipe.id)
    assert created is not None  # noqa: S101
    return created


async def upsert_external_recipe(db: AsyncSession, data: ExternalRecipe) -> tuple[Recipe, bool]:
    """Insert or update an imported recipe identified by `(source, external_id)`.

    Ingredients are matched (or created) by normalized name; when the source lists an
    ingredient twice, the first occurrence wins. The change is committed.

    Args:
        db: Database session.
        data: The recipe as read from the external source.

    Returns:
        `(recipe, created)`; the recipe is re-loaded with ingredients and chunks.
    """
    existing = await db.scalars(
        select(Recipe)
        .where(Recipe.source == data.source, Recipe.external_id == data.external_id)
        .options(_WITH_INGREDIENTS)
    )
    recipe = existing.one_or_none()
    created = recipe is None
    if recipe is None:
        recipe = Recipe(source=data.source, external_id=data.external_id, ingredients=[])
        db.add(recipe)

    recipe.title = data.title
    recipe.instructions = data.instructions
    recipe.category = data.category
    recipe.area = data.area
    recipe.tags = list(data.tags)
    recipe.thumbnail_url = data.thumbnail_url
    recipe.source_url = data.source_url

    ingredients = await get_or_create_ingredients(db, (item.name for item in data.ingredients))
    lines: dict[int, tuple[float | None, str | None]] = {}
    for item in data.ingredients:
        ingredient = ingredients.get(normalize_ingredient_name(item.name))
        if ingredient is not None:
            lines.setdefault(ingredient.id, (item.quantity, item.unit))

    current = {link.ingredient_id: (link.quantity, link.unit) for link in recipe.ingredients}
    if created or current != lines:
        # Same (recipe_id, ingredient_id) keys become UPDATEs ("row switch"), others INSERT/DELETE.
        recipe.ingredients = [
            RecipeIngredient(ingredient_id=ingredient_id, quantity=quantity, unit=unit)
            for ingredient_id, (quantity, unit) in lines.items()
        ]

    await db.commit()
    reloaded = await get_recipe(db, recipe.id, with_chunks=True)
    assert reloaded is not None  # noqa: S101
    return reloaded, created
