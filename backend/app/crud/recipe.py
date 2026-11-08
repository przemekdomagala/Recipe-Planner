from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import IngredientsNotFoundError
from app.models import Ingredient, Recipe, RecipeIngredient
from app.schemas import RecipeCreate

_WITH_INGREDIENTS = selectinload(Recipe.ingredients).selectinload(RecipeIngredient.ingredient)


async def get_recipe(db: AsyncSession, recipe_id: int) -> Recipe | None:
    result = await db.scalars(
        select(Recipe)
        .where(Recipe.id == recipe_id)
        .options(_WITH_INGREDIENTS)
        .execution_options(populate_existing=True)
    )
    return result.one_or_none()


async def get_recipes(db: AsyncSession, *, skip: int = 0, limit: int = 100) -> Sequence[Recipe]:
    result = await db.scalars(
        select(Recipe).order_by(Recipe.id).offset(skip).limit(limit).options(_WITH_INGREDIENTS)
    )
    return result.all()


async def create_recipe(db: AsyncSession, data: RecipeCreate) -> Recipe:
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
