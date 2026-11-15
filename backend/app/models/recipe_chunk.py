"""Recipe chunk ORM model: the vector store of the RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base

if TYPE_CHECKING:
    from app.models.recipe import Recipe


class RecipeChunk(Base):
    """A retrievable piece of a recipe's text together with its embedding.

    Chunks are derived data: they are rebuilt from the recipe by `app.rag.indexing` and
    can be dropped and regenerated at any time (e.g. after switching embedding models).
    """

    __tablename__ = "recipe_chunks"
    __table_args__ = (
        UniqueConstraint("recipe_id", "chunk_index", name="uq_recipe_chunks_recipe_index"),
        # Approximate nearest-neighbour index for cosine distance; PostgreSQL only.
        Index(
            "ix_recipe_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ).ddl_if(dialect="postgresql"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int]
    # A `ChunkKind` value ("summary" or "instructions").
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(100))
    # A pgvector `vector(dim)` on PostgreSQL; falls back to JSON on SQLite so the fast unit
    # tests can run without a database server.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dim).with_variant(JSON(), "sqlite")
    )

    recipe: Mapped[Recipe] = relationship()
