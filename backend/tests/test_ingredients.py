"""Tests for the ingredient endpoints."""

from httpx import AsyncClient


async def test_create_ingredient(client: AsyncClient) -> None:
    """Creating an ingredient returns 201 and the normalized name."""
    response = await client.post("/api/ingredients", json={"name": "Tomato"})

    assert response.status_code == 201
    body = response.json()
    assert body["id"] > 0
    assert body["name"] == "tomato"  # normalized


async def test_create_ingredient_duplicate_returns_409(client: AsyncClient) -> None:
    """A name equal to an existing one after normalization is rejected with 409."""
    assert (await client.post("/api/ingredients", json={"name": "Basil"})).status_code == 201

    response = await client.post("/api/ingredients", json={"name": "  basil "})

    assert response.status_code == 409


async def test_create_ingredient_rejects_blank_name(client: AsyncClient) -> None:
    """A whitespace-only name fails validation."""
    response = await client.post("/api/ingredients", json={"name": "   "})

    assert response.status_code == 422


async def test_list_ingredients_sorted_by_name(client: AsyncClient) -> None:
    """Ingredients are listed alphabetically."""
    for name in ("pasta", "garlic", "onion"):
        await client.post("/api/ingredients", json={"name": name})

    response = await client.get("/api/ingredients")

    assert response.status_code == 200
    assert [item["name"] for item in response.json()] == ["garlic", "onion", "pasta"]
