from httpx import AsyncClient


async def test_search_placeholder_returns_mock_response(client: AsyncClient) -> None:
    response = await client.post("/api/recipes/search", json={"query": "quick vegan dinner"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "quick vegan dinner"
    assert body["mock"] is True
    assert isinstance(body["results"], list)


async def test_search_rejects_empty_query(client: AsyncClient) -> None:
    response = await client.post("/api/recipes/search", json={"query": ""})

    assert response.status_code == 422


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
