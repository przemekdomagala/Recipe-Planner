"""Database access for recipes and their ingredient links."""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import IngredientsNotFoundError
from app.models import Ingredient, Recipe, RecipeIngredient
from app.schemas import RecipeCreate

_WITH_INGREDIENTS = selectinload(Recipe.ingredients).selectinload(RecipeIngredient.ingredient)


async def get_recipe(db: AsyncSession, recipe_id: int) -> Recipe | None:
    """Return a recipe with its ingredients eagerly loaded, or `None` if missing.

    `populate_existing` refreshes an instance already in the session's identity map, so
    relationships added in the same session are reflected.
    """
    result = await db.scalars(
        select(Recipe)
        .where(Recipe.id == recipe_id)
        .options(_WITH_INGREDIENTS)
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
