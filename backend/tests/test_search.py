"""Tests for semantic recipe search (`POST /api/recipes/search`)."""

from app.schemas import ExternalRecipe
from app.services.ingestion import ingest_recipes
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeEmbedder

RECIPES = [
    ("1", "Spaghetti Carbonara", "Pasta", "Italian", "Boil spaghetti. Mix eggs and pecorino."),
    ("2", "Beef Rendang", "Beef", "Malaysian", "Slow cook beef in coconut milk and spices."),
    ("3", "Margherita Pizza", "Vegetarian", "Italian", "Bake dough with tomato and mozzarella."),
]


async def seed(db: AsyncSession, embedder: FakeEmbedder) -> None:
    """Ingest a few small recipes."""
    recipes = [
        ExternalRecipe(
            source="test",
            external_id=external_id,
            title=title,
            category=category,
            area=area,
            instructions=instructions,
        )
        for external_id, title, category, area, instructions in RECIPES
    ]
    await ingest_recipes(db, recipes, embedder)


async def test_search_ranks_relevant_recipe_first(
    client: AsyncClient, db_session: AsyncSession, embedder: FakeEmbedder
) -> None:
    """The recipe sharing the query's words ranks first, with its matching chunks."""
    await seed(db_session, embedder)

    response = await client.post(
        "/api/recipes/search", json={"query": "slow cook beef with coconut milk", "limit": 2}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "slow cook beef with coconut milk"
    assert len(body["results"]) == 2
    top = body["results"][0]
    assert top["title"] == "Beef Rendang"
    assert top["area"] == "Malaysian"
    assert top["matches"][0]["kind"] in {"overview", "instructions"}
    assert top["score"] >= body["results"][1]["score"]
    assert len({hit["recipe_id"] for hit in body["results"]}) == 2  # one hit per recipe


async def test_search_filters_and_min_score(
    client: AsyncClient, db_session: AsyncSession, embedder: FakeEmbedder
) -> None:
    """Metadata filters are case-insensitive; `min_score` drops weak hits."""
    await seed(db_session, embedder)

    italian = await client.post(
        "/api/recipes/search", json={"query": "beef", "filters": {"area": "italian"}}
    )
    strict = await client.post(
        "/api/recipes/search", json={"query": "spaghetti carbonara", "min_score": 0.3}
    )

    assert {hit["title"] for hit in italian.json()["results"]} == {
        "Spaghetti Carbonara",
        "Margherita Pizza",
    }
    assert [hit["title"] for hit in strict.json()["results"]] == ["Spaghetti Carbonara"]


async def test_search_on_empty_knowledge_base(client: AsyncClient) -> None:
    """With nothing indexed, search returns no results rather than failing."""
    response = await client.post("/api/recipes/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_rejects_blank_query(client: AsyncClient) -> None:
    """An empty or whitespace-only query fails validation."""
    for query in ("", "   "):
        response = await client.post("/api/recipes/search", json={"query": query})
        assert response.status_code == 422
