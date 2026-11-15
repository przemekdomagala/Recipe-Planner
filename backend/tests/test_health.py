"""Tests for the health endpoint."""

from httpx import AsyncClient


async def test_health(client: AsyncClient) -> None:
    """The health endpoint reports `ok`."""
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
