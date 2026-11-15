"""Tests for the health endpoints."""

import httpx
from app.api.deps import get_chat_client, get_embedding_client
from app.clients.ollama import OllamaClient
from app.main import create_app
from httpx import ASGITransport, AsyncClient


async def test_health(client: AsyncClient) -> None:
    """The health endpoint reports `ok`."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_llm_health_reports_models() -> None:
    """`/health/llm` lists reachability and which configured models are installed."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Pretend only the embedding model is available."""
        return httpx.Response(200, json={"models": [{"name": "nomic-embed-text:latest"}]})

    cloud_handler = lambda request: httpx.Response(  # noqa: E731
        200, json={"models": [{"name": "gemma4:31b"}]}
    )

    ollama = OllamaClient(
        base_url="http://ollama",
        chat_model="gemma4:31b",
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        transport=httpx.MockTransport(handler),
    )
    app = create_app()
    app.dependency_overrides[get_chat_client] = lambda: ollama
    app.dependency_overrides[get_embedding_client] = lambda: ollama
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        body = (await client.get("/health/llm")).json()

    assert body["chat"]["reachable"] is True
    assert body["chat"]["chat_model_available"] is False
    assert body["embeddings"]["embedding_model_available"] is True

    cloud = OllamaClient(
        base_url="https://ollama.com",
        chat_model="gemma4:31b-cloud",
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        transport=httpx.MockTransport(cloud_handler),
    )
    async with cloud:
        assert (await cloud.status()).chat_model_available is True  # "-cloud" == direct name
