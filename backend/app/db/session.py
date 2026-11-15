"""Async engine, session factory, start-up schema bootstrap and the `get_db` dependency."""

from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import app.models  # noqa: F401  (registers the models on Base.metadata for create_all)
from app.core.config import settings
from app.db.base import Base

# pgvector values travel in their text form ("[0.1,0.2,...]"), which is what
# `pgvector.sqlalchemy.Vector` binds. Do not register pgvector's binary asyncpg codec
# (`pgvector.asyncpg.register_vector`): it rejects those strings, breaking vector queries.
engine: AsyncEngine = create_async_engine(settings.database_url, echo=settings.sql_echo)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Enable the pgvector extension and create all tables.

    This is a bootstrap convenience; a migration tool such as Alembic should replace
    `create_all` once the schema starts to evolve.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a database session scoped to a single request."""
    async with AsyncSessionLocal() as session:
        yield session
