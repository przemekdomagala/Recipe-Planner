"""Database access for ingredients."""

from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateIngredientError
from app.models import Ingredient
from app.schemas import IngredientCreate, normalize_ingredient_name


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


async def get_or_create_ingredients(
    db: AsyncSession, names: Iterable[str]
) -> dict[str, Ingredient]:
    """Return ingredients by name, inserting the missing ones (without committing).

    Args:
        db: Database session.
        names: Ingredient names; they are normalized like `IngredientCreate` does.

    Returns:
        Mapping from normalized name to ingredient (new ones are flushed, so have ids).
    """
    wanted = {normalize_ingredient_name(name) for name in names} - {""}
    if not wanted:
        return {}
    existing = await db.scalars(select(Ingredient).where(Ingredient.name.in_(wanted)))
    by_name = {ingredient.name: ingredient for ingredient in existing}
    missing = [Ingredient(name=name) for name in sorted(wanted - by_name.keys())]
    if missing:
        db.add_all(missing)
        await db.flush()
        by_name.update((ingredient.name, ingredient) for ingredient in missing)
    return by_name
