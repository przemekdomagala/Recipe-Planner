from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateIngredientError
from app.models import Ingredient
from app.schemas import IngredientCreate


async def get_ingredients(
    db: AsyncSession, *, skip: int = 0, limit: int = 100
) -> Sequence[Ingredient]:
    result = await db.scalars(
        select(Ingredient).order_by(Ingredient.name).offset(skip).limit(limit)
    )
    return result.all()


async def create_ingredient(db: AsyncSession, data: IngredientCreate) -> Ingredient:
    ingredient = Ingredient(name=data.name)
    db.add(ingredient)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateIngredientError(data.name) from exc
    return ingredient
