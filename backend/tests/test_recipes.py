"""Tests for the recipe CRUD endpoints and indexing on creation."""

from app.models import RecipeChunk
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeEmbedder


async def _create_ingredient(client: AsyncClient, name: str) -> int:
    """Create an ingredient through the API and return its id."""
    response = await client.post("/api/ingredients", json={"name": name})
    assert response.status_code == 201
    ingredient_id: int = response.json()["id"]
    return ingredient_id


async def test_create_recipe_with_ingredients(client: AsyncClient) -> None:
    """A recipe is created with its ingredient lines in the given order."""
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
    assert [
        (item["ingredient"]["name"], item["quantity"], item["unit"]) for item in body["ingredients"]
    ] == [("pasta", 200.0, "g"), ("garlic", 3.0, "cloves")]


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


async def test_created_recipe_has_user_source_defaults(client: AsyncClient) -> None:
    """Recipes created through the API are marked as user recipes without metadata."""
    body = (
        await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})
    ).json()

    assert body["source"] == "user"
    assert body["external_id"] is None
    assert body["tags"] == []
    assert body["category"] is None


async def test_create_recipe_indexes_it(client: AsyncClient, db_session: AsyncSession) -> None:
    """A new recipe is chunked and embedded right away."""
    created = (
        await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})
    ).json()

    chunks = await db_session.scalars(
        select(RecipeChunk).where(RecipeChunk.recipe_id == created["id"])
    )
    assert [chunk.kind for chunk in chunks] == ["summary", "instructions"]


async def test_create_recipe_succeeds_when_embedder_is_down(
    client: AsyncClient, db_session: AsyncSession, fake_embedder: FakeEmbedder
) -> None:
    """Indexing is best-effort: the recipe is created but has no chunks."""
    fake_embedder.fail = True

    response = await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})

    assert response.status_code == 201
    assert await db_session.scalar(select(func.count()).select_from(RecipeChunk)) == 0
