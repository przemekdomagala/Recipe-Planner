"""Recipe ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.recipe_ingredient import RecipeIngredient

USER_SOURCE = "user"
"""`Recipe.source` of recipes created through the API rather than imported."""


class Recipe(Base):
    """A recipe with free-text instructions, descriptive metadata and ingredient links.

    Imported recipes are identified by `(source, external_id)`, which makes re-imports
    update the existing row instead of duplicating it. Vector embeddings live in
    `RecipeChunk`, not here.
    """

    __tablename__ = "recipes"
    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_recipes_source_external"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    instructions: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default=USER_SOURCE, index=True)
    external_id: Mapped[str | None] = mapped_column(String(64), default=None)
    category: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    area: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    image_url: Mapped[str | None] = mapped_column(String(500), default=None)
    source_url: Mapped[str | None] = mapped_column(String(1000), default=None)

    ingredients: Mapped[list[RecipeIngredient]] = relationship(
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.position",
    )
