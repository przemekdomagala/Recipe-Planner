"""Recipe chunk ORM model: the unit of retrieval in the RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.recipe import Recipe


class RecipeChunk(Base):
    """A piece of a recipe's text together with its embedding.

    Each recipe is split into an `overview` chunk (title, category, cuisine, tags and
    ingredients) and one or more `instructions` chunks; see `app.services.chunking`.
    """

    __tablename__ = "recipe_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id", ondelete="CASCADE"), index=True)
    position: Mapped[int]
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    # Model that produced `embedding`; vectors from different models aren't comparable,
    # so a model change makes the chunk stale (see `app.services.indexing`).
    embedding_model: Mapped[str] = mapped_column(String(100))
    # pgvector `vector(dim)` on PostgreSQL, JSON on SQLite (test suite). Deferred with
    # raiseload so the large vectors are only loaded when selected explicitly.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dim).with_variant(JSON(), "sqlite"),
        deferred=True,
        deferred_raiseload=True,
    )

    recipe: Mapped[Recipe] = relationship(back_populates="chunks")

    __table_args__ = (
        # Approximate nearest-neighbour index for cosine distance (PostgreSQL only).
        Index(
            "ix_recipe_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ).ddl_if(dialect="postgresql"),
    )
