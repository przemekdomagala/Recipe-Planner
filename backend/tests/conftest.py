"""Shared fixtures.

By default the tests run against an in-memory SQLite database (no services required). 
Setting the `TEST_DATABASE_URL` environment variable (e.g. `postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test`) will run
the tests against a real PostgreSQL + pgvector instance.
"""

import os
from collections.abc import AsyncIterator

import app.models  # noqa: F401  (registers the models on Base.metadata)
import pytest
from app.db.base import Base
from app.db.session import get_db
from app.main import create_app
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    if TEST_DATABASE_URL.startswith("sqlite"):
        # StaticPool keeps one shared connection, so the in-memory DB survives across sessions.
        test_engine = create_async_engine(
            TEST_DATABASE_URL, poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    else:
        test_engine = create_async_engine(TEST_DATABASE_URL)

    async with test_engine.begin() as conn:
        if test_engine.dialect.name == "postgresql":
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture
async def db_session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture
async def client(engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    """HTTP client wired to the app, with `get_db` overridden to use the test database."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client
