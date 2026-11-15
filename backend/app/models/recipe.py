"""Recipe ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.recipe_chunk import RecipeChunk
    from app.models.recipe_ingredient import RecipeIngredient

USER_SOURCE = "user"


class Recipe(Base):
    """A recipe with free-text instructions, ingredient links and its retrieval chunks.

    Recipes are either created through the API (`source="user"`) or imported from an
    external dataset, identified there by `(source, external_id)`.
    """

    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    instructions: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default=USER_SOURCE)
    external_id: Mapped[str | None] = mapped_column(String(64), default=None)
    category: Mapped[str | None] = mapped_column(String(64), default=None)
    area: Mapped[str | None] = mapped_column(String(64), default=None)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    thumbnail_url: Mapped[str | None] = mapped_column(String(500), default=None)
    source_url: Mapped[str | None] = mapped_column(String(500), default=None)

    ingredients: Mapped[list[RecipeIngredient]] = relationship(
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.ingredient_id",
    )
    chunks: Mapped[list[RecipeChunk]] = relationship(
        back_populates="recipe",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="RecipeChunk.position",
    )

    __table_args__ = (UniqueConstraint("source", "external_id", name="uq_recipes_source_external"),)
