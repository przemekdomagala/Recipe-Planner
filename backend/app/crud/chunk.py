"""Vector store access: nearest-neighbour search over recipe chunks."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import ColumnElement, String, cast, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager

from app.models import Recipe, RecipeChunk
from app.schemas import RecipeSearchFilters


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk (with its recipe loaded) and its cosine similarity to the query."""

    chunk: RecipeChunk
    score: float


def _filter_clauses(filters: RecipeSearchFilters | None) -> list[ColumnElement[bool]]:
    """Case-insensitive metadata filters on the chunk's recipe."""
    clauses: list[ColumnElement[bool]] = []
    if filters and filters.category:
        clauses.append(func.lower(Recipe.category) == filters.category.lower())
    if filters and filters.area:
        clauses.append(func.lower(Recipe.area) == filters.area.lower())
    return clauses


async def nearest_chunks(
    db: AsyncSession,
    query_vector: Sequence[float],
    *,
    k: int,
    filters: RecipeSearchFilters | None = None,
) -> list[ScoredChunk]:
    """Return the `k` chunks most similar to `query_vector`, best first.

    On PostgreSQL this is a pgvector cosine-distance query (served by the HNSW index). On
    other databases (the SQLite test suite) similarities are computed in Python with numpy,
    which is exact but loads every candidate vector, so it's only meant for small data.

    Args:
        db: Database session.
        query_vector: The embedded query.
        k: Maximum number of chunks to return.
        filters: Optional recipe metadata filters.
    """
    base = (
        select(RecipeChunk)
        .join(RecipeChunk.recipe)
        .options(contains_eager(RecipeChunk.recipe))
        .where(*_filter_clauses(filters))
    )

    if db.get_bind().dialect.name == "postgresql":
        # Bound as VARCHAR and cast in SQL: pgvector's SQLAlchemy type always sends text,
        # which the binary asyncpg codec registered in `db.session` would reject.
        query_text = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        query_param = cast(literal(query_text, String()), Vector(len(query_vector)))
        distance = RecipeChunk.embedding.cosine_distance(query_param)
        rows = await db.execute(
            base.add_columns(distance.label("distance")).order_by(distance).limit(k)
        )
        return [ScoredChunk(chunk, 1.0 - float(dist)) for chunk, dist in rows]

    candidates = (
        await db.execute(
            select(RecipeChunk.id, RecipeChunk.embedding)
            .join(RecipeChunk.recipe)
            .where(*_filter_clauses(filters))
        )
    ).all()
    if not candidates:
        return []
    matrix = np.asarray([vector for _, vector in candidates], dtype=np.float64)
    query = np.asarray(query_vector, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(matrix @ query, norms, out=np.zeros(len(candidates)), where=norms > 0)
    top = np.argsort(-scores)[:k]
    score_by_id = {int(candidates[i][0]): float(scores[i]) for i in top}

    chunks = await db.scalars(base.where(RecipeChunk.id.in_(score_by_id)))
    ranked = sorted(chunks, key=lambda chunk: score_by_id[chunk.id], reverse=True)
    return [ScoredChunk(chunk, score_by_id[chunk.id]) for chunk in ranked]
