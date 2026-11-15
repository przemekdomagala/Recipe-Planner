"""Shared fixtures.

By default the tests run against an in-memory SQLite database (no services required).
Setting the `TEST_DATABASE_URL` environment variable (e.g.
`postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test`) runs the tests against a
real PostgreSQL + pgvector instance.
"""

import os
from collections.abc import AsyncIterator

import app.models  # noqa: F401  (registers the models on Base.metadata)
import pytest
from app.api.deps import get_chat_model, get_embedder
from app.db.base import Base
from app.db.session import _register_vector_codec, get_db
from app.main import create_app
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from tests.fakes import FakeChatModel, FakeEmbedder

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    """Provide an engine on a freshly created schema, dropped again after the test."""
    if TEST_DATABASE_URL.startswith("sqlite"):
        # StaticPool keeps one shared connection, so the in-memory DB survives across sessions.
        test_engine = create_async_engine(
            TEST_DATABASE_URL, poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    else:
        test_engine = create_async_engine(TEST_DATABASE_URL)
        # Mirror the app engine: the binary pgvector codec changes how vectors are bound.
        event.listen(test_engine.sync_engine, "connect", _register_vector_codec)

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
    """Provide a session on the test database for direct ORM access in tests."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest.fixture
def embedder() -> FakeEmbedder:
    """Deterministic embedder shared by the app and the test."""
    return FakeEmbedder()


@pytest.fixture
def chat_model() -> FakeChatModel:
    """Fake chat model shared by the app and the test."""
    return FakeChatModel()


@pytest.fixture
async def client(
    engine: AsyncEngine, embedder: FakeEmbedder, chat_model: FakeChatModel
) -> AsyncIterator[AsyncClient]:
    """HTTP client wired to the app, using the test database and the fake models."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        """Yield a session bound to the test engine instead of the app's engine."""
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client
