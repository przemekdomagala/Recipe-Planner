"""Database access for ingredients."""

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
    """Return ingredients ordered by name.

    Args:
        db: Database session.
        skip: Number of rows to skip.
        limit: Maximum number of rows to return.
    """
    result = await db.scalars(
        select(Ingredient).order_by(Ingredient.name).offset(skip).limit(limit)
    )
    return result.all()


async def create_ingredient(db: AsyncSession, data: IngredientCreate) -> Ingredient:
    """Insert a new ingredient.

    Args:
        db: Database session.
        data: Validated (already normalized) ingredient payload.

    Returns:
        The persisted ingredient.

    Raises:
        DuplicateIngredientError: If the name violates the unique constraint.
    """
    ingredient = Ingredient(name=data.name)
    db.add(ingredient)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise DuplicateIngredientError(data.name) from exc
    return ingredient
