from __future__ import annotations

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.recipe_ingredient import RecipeIngredient


class Recipe(Base):
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    instructions: Mapped[str] = mapped_column(Text)
    # Reserved for the RAG pipeline: stays NULL until an embedding is computed for the recipe.
    # A pgvector `vector(dim)` on PostgreSQL; falls back to JSON on SQLite so the fast unit
    # tests can run without a database server.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dim).with_variant(JSON(), "sqlite"), default=None
    )

    ingredients: Mapped[list[RecipeIngredient]] = relationship(
        back_populates="recipe",
        cascade="all, delete-orphan",
        order_by="RecipeIngredient.ingredient_id",
    )
