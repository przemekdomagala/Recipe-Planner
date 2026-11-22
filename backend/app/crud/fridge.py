"""Database access for fridge items."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateFridgeItemError
from app.crud.ingredient import get_or_create_ingredients
from app.models import FridgeItem
from app.schemas import FridgeItemCreate, FridgeItemUpdate


async def get_fridge_items(db: AsyncSession) -> list[FridgeItem]:
    """Return all fridge items, soonest expiry first, then items without a date, by name.

    Args:
        db: Database session.

    Returns:
        The items with their ingredients loaded.
    """
    items = list(await db.scalars(select(FridgeItem)))
    return sorted(
        items,
        key=lambda item: (item.expires_on is None, item.expires_on, item.ingredient.name),
    )


async def get_fridge_item(db: AsyncSession, item_id: int) -> FridgeItem | None:
    """Return one fridge item with its ingredient, or `None` if it doesn't exist."""
    return await db.get(FridgeItem, item_id)


async def add_fridge_item(db: AsyncSession, data: FridgeItemCreate) -> FridgeItem:
    """Put an ingredient into the fridge, creating the ingredient if it's new.

    Args:
        db: Database session.
        data: Validated payload; the name is already normalized.

    Returns:
        The persisted item with its ingredient loaded.

    Raises:
        DuplicateFridgeItemError: If the ingredient is already in the fridge.
    """
    ingredient = (await get_or_create_ingredients(db, [data.name]))[data.name]
    existing = await db.scalar(select(FridgeItem).where(FridgeItem.ingredient_id == ingredient.id))
    if existing is not None:
        await db.rollback()
        raise DuplicateFridgeItemError(data.name)
    item = FridgeItem(
        ingredient=ingredient, quantity=data.quantity, unit=data.unit, expires_on=data.expires_on
    )
    db.add(item)
    await db.commit()
    return item


async def update_fridge_item(
    db: AsyncSession, item: FridgeItem, data: FridgeItemUpdate
) -> FridgeItem:
    """Apply the fields set in `data` to `item` and commit.

    Args:
        db: Database session.
        item: The item to change.
        data: Partial update; fields that weren't sent are left as they are.

    Returns:
        The updated item.
    """
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(item, field, value)
    await db.commit()
    return item


async def delete_fridge_item(db: AsyncSession, item: FridgeItem) -> None:
    """Remove an item from the fridge (the ingredient itself is kept)."""
    await db.delete(item)
    await db.commit()
