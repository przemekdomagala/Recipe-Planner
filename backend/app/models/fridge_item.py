"""Fridge item ORM model: an ingredient the user currently has at home."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.ingredient import Ingredient


class FridgeItem(Base):
    """An ingredient in the user's fridge, with an optional amount and expiry date.

    Each ingredient appears at most once; `expires_on` drives the waste-minimising ranking
    of fridge suggestions.
    """

    __tablename__ = "fridge_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="CASCADE"), unique=True
    )
    quantity: Mapped[float | None] = mapped_column(default=None)
    unit: Mapped[str | None] = mapped_column(String(32), default=None)
    expires_on: Mapped[date | None] = mapped_column(default=None)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    ingredient: Mapped[Ingredient] = relationship(lazy="joined")
