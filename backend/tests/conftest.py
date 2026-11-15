"""Shared fixtures.

By default the tests run against an in-memory SQLite database (no services required).
Setting the `TEST_DATABASE_URL` environment variable (e.g.
`postgresql+asyncpg://recipe:recipe@localhost:5433/recipe_test`) runs the tests against a
real PostgreSQL + pgvector instance, which also enables the tests marked `postgres`
(vector similarity search cannot run on SQLite).

The LLM is never called: the embedding and chat models are replaced by the fakes in
`tests.fakes`, and the Ollama server by an `httpx.MockTransport`.
"""

import os
from collections.abc import AsyncIterator

import app.models  # noqa: F401  (registers the models on Base.metadata)
import httpx
import pytest
from app.api.deps import get_chat_model, get_embedder, get_ollama
from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.llm import Ollama
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

from tests.fakes import FakeChatModel, FakeEmbedder

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
USES_POSTGRES = TEST_DATABASE_URL.startswith("postgresql")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests marked `postgres` unless the suite runs against PostgreSQL."""
    if USES_POSTGRES:
        return
    skip = pytest.mark.skip(reason="needs PostgreSQL + pgvector (set TEST_DATABASE_URL)")
    for item in items:
        if "postgres" in item.keywords:
            item.add_marker(skip)


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
def fake_embedder() -> FakeEmbedder:
    """Deterministic embedder shared by the app and the test."""
    return FakeEmbedder()


@pytest.fixture
def fake_chat_model() -> FakeChatModel:
    """Recording chat model shared by the app and the test."""
    return FakeChatModel()


@pytest.fixture
def ollama_models() -> list[str]:
    """Model names the mocked Ollama server reports as pulled; tests may mutate it."""
    return [f"{settings.ollama_embedding_model}:latest", settings.ollama_chat_model]


@pytest.fixture
async def ollama(ollama_models: list[str]) -> AsyncIterator[Ollama]:
    """Ollama facade talking to a mock server that only implements `GET /api/tags`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the model list; anything else is a test bug."""
        assert request.url.path == "/api/tags", f"unexpected Ollama call {request.url}"
        return httpx.Response(200, json={"models": [{"name": name} for name in ollama_models]})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama.test"
    ) as http:
        yield Ollama.from_settings(http, settings)


@pytest.fixture
async def client(
    engine: AsyncEngine,
    fake_embedder: FakeEmbedder,
    fake_chat_model: FakeChatModel,
    ollama: Ollama,
) -> AsyncIterator[AsyncClient]:
    """HTTP client wired to the app, with the database and LLM dependencies overridden."""
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        """Yield a session bound to the test engine instead of the app's engine."""
        async with session_factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama] = lambda: ollama
    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_chat_model] = lambda: fake_chat_model
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        yield http_client
