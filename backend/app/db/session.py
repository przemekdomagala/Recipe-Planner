"""Async engine, session factory, start-up schema bootstrap and the `get_db` dependency."""

import contextlib
from collections.abc import AsyncIterator

from pgvector.asyncpg import register_vector
from sqlalchemy import AdaptedConnection, event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import ConnectionPoolEntry

import app.models  # noqa: F401  (registers the models on Base.metadata for create_all)
from app.core.config import settings
from app.db.base import Base

engine: AsyncEngine = create_async_engine(settings.database_url, echo=settings.sql_echo)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _register_vector_codec(
    dbapi_connection: AdaptedConnection, _connection_record: ConnectionPoolEntry
) -> None:
    """Teach asyncpg how to encode/decode pgvector values on every new connection."""
    # ValueError means the `vector` extension doesn't exist yet (first start-up). `init_db`
    # creates it and then disposes the pool so that fresh connections register the codec.
    with contextlib.suppress(ValueError):
        dbapi_connection.run_async(register_vector)


async def init_db() -> None:
    """Enable the pgvector extension and create all tables.

    This is a bootstrap convenience; a migration tool such as Alembic should replace
    `create_all` once the schema starts to evolve.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    # Drop connections opened before the extension existed so new ones register the codec.
    await engine.dispose()


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a database session scoped to a single request."""
    async with AsyncSessionLocal() as session:
        yield session
