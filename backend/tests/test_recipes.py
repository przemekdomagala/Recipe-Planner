"""Tests for the recipe endpoints, including indexing on create."""

from collections.abc import Sequence

from app.core.config import settings
from app.core.exceptions import LLMUnavailableError
from app.models import RecipeChunk
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from tests.fakes import FakeEmbedder


async def _create_ingredient(client: AsyncClient, name: str) -> int:
    """Create an ingredient through the API and return its id."""
    response = await client.post("/api/ingredients", json={"name": name})
    assert response.status_code == 201
    ingredient_id: int = response.json()["id"]
    return ingredient_id


async def test_create_recipe_with_ingredients(client: AsyncClient) -> None:
    """A recipe is created with its ingredient lines; the embedding is not exposed."""
    pasta_id = await _create_ingredient(client, "pasta")
    garlic_id = await _create_ingredient(client, "garlic")

    response = await client.post(
        "/api/recipes",
        json={
            "title": "Aglio e olio",
            "instructions": "Boil pasta. Fry garlic. Combine.",
            "ingredients": [
                {"ingredient_id": pasta_id, "quantity": 200, "unit": "g"},
                {"ingredient_id": garlic_id, "quantity": 3, "unit": "cloves"},
            ],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Aglio e olio"
    assert "embedding" not in body
    assert {
        (item["ingredient"]["name"], item["quantity"], item["unit"]) for item in body["ingredients"]
    } == {("pasta", 200.0, "g"), ("garlic", 3.0, "cloves")}


async def test_create_recipe_without_ingredients(client: AsyncClient) -> None:
    """Ingredients are optional when creating a recipe."""
    response = await client.post(
        "/api/recipes", json={"title": "Water", "instructions": "Pour into a glass."}
    )

    assert response.status_code == 201
    assert response.json()["ingredients"] == []


async def test_create_recipe_unknown_ingredient_returns_422(client: AsyncClient) -> None:
    """Unknown ingredient ids are reported with 422 and nothing is persisted."""
    response = await client.post(
        "/api/recipes",
        json={"title": "Ghost", "instructions": "Boo.", "ingredients": [{"ingredient_id": 999}]},
    )

    assert response.status_code == 422
    assert "999" in response.json()["detail"]
    assert (await client.get("/api/recipes")).json() == []  # nothing was persisted


async def test_create_recipe_duplicate_ingredient_returns_422(client: AsyncClient) -> None:
    """Listing the same ingredient twice fails validation."""
    ingredient_id = await _create_ingredient(client, "salt")

    response = await client.post(
        "/api/recipes",
        json={
            "title": "Salty",
            "instructions": "Add salt.",
            "ingredients": [{"ingredient_id": ingredient_id}, {"ingredient_id": ingredient_id}],
        },
    )

    assert response.status_code == 422


async def test_create_recipe_validates_required_fields(client: AsyncClient) -> None:
    """An empty title fails validation."""
    response = await client.post("/api/recipes", json={"title": "", "instructions": "x"})

    assert response.status_code == 422


async def test_list_recipes_empty(client: AsyncClient) -> None:
    """Listing recipes on an empty database returns an empty list."""
    response = await client.get("/api/recipes")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_recipes_returns_created_recipes_with_pagination(client: AsyncClient) -> None:
    """Recipes are listed in creation order and `skip` / `limit` paginate them."""
    for title in ("First", "Second", "Third"):
        await client.post("/api/recipes", json={"title": title, "instructions": "Cook."})

    all_recipes = await client.get("/api/recipes")
    page = await client.get("/api/recipes", params={"skip": 1, "limit": 1})

    assert [r["title"] for r in all_recipes.json()] == ["First", "Second", "Third"]
    assert [r["title"] for r in page.json()] == ["Second"]


async def test_get_recipe_by_id(client: AsyncClient) -> None:
    """An existing recipe is returned by id; an unknown id gives 404."""
    created = (
        await client.post("/api/recipes", json={"title": "Toast", "instructions": "Toast bread."})
    ).json()

    found = await client.get(f"/api/recipes/{created['id']}")
    missing = await client.get("/api/recipes/12345")

    assert found.status_code == 200
    assert found.json()["title"] == "Toast"
    assert missing.status_code == 404


async def test_create_recipe_stores_metadata(client: AsyncClient) -> None:
    """Category, cuisine and tags are cleaned and returned; source defaults to "user"."""
    response = await client.post(
        "/api/recipes",
        json={
            "title": "Pho",
            "instructions": "Simmer broth.",
            "category": " Soup ",
            "area": "",
            "tags": ["noodles", " noodles", "", "spicy"],
        },
    )

    body = response.json()
    assert (body["source"], body["category"], body["area"]) == ("user", "Soup", None)
    assert body["tags"] == ["noodles", "spicy"]


async def test_created_recipe_is_indexed(client: AsyncClient, db_session: AsyncSession) -> None:
    """Creating a recipe builds its overview and instructions chunks with embeddings."""
    created = (
        await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})
    ).json()

    chunks = (
        await db_session.scalars(
            select(RecipeChunk)
            .where(RecipeChunk.recipe_id == created["id"])
            .order_by(RecipeChunk.position)
            .options(undefer(RecipeChunk.embedding))
        )
    ).all()
    assert [chunk.kind for chunk in chunks] == ["overview", "instructions"]
    assert all(len(chunk.embedding) == settings.embedding_dim for chunk in chunks)
    assert {chunk.embedding_model for chunk in chunks} == {"fake-embed"}


async def test_recipe_is_created_even_if_indexing_fails(
    client: AsyncClient, embedder: FakeEmbedder, db_session: AsyncSession
) -> None:
    """An unavailable embedding model doesn't block recipe creation."""

    async def unavailable(texts: Sequence[str]) -> list[list[float]]:
        """Simulate Ollama being down."""
        raise LLMUnavailableError("down")

    embedder.embed_documents = unavailable  # type: ignore[method-assign]

    response = await client.post("/api/recipes", json={"title": "Stew", "instructions": "Stew."})

    assert response.status_code == 201
    assert (await db_session.scalars(select(RecipeChunk))).all() == []
