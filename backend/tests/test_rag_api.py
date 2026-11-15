"""Tests for semantic search (`/api/recipes/search`) and the RAG endpoints (`/api/rag`)."""

import pytest
from app.core.config import settings
from app.rag import index_recipe
from app.services import import_recipe
from app.sources import ExternalIngredient, ExternalRecipe
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeChatModel, FakeEmbedder

KNOWLEDGE_BASE = [
    ExternalRecipe(
        source="themealdb",
        external_id="1",
        title="Chicken Curry",
        instructions="Fry onion and garlic. Add chicken and curry paste. Simmer with rice.",
        category="Chicken",
        area="Indian",
        ingredients=(ExternalIngredient("Chicken", "500g"), ExternalIngredient("Rice", "1 cup")),
    ),
    ExternalRecipe(
        source="themealdb",
        external_id="2",
        title="Chocolate Cake",
        instructions="Mix flour, cocoa, sugar and eggs. Bake for forty minutes.",
        category="Dessert",
        area="British",
        ingredients=(ExternalIngredient("Cocoa", "50g"), ExternalIngredient("Sugar", "200g")),
    ),
    ExternalRecipe(
        source="themealdb",
        external_id="3",
        title="Vegetable Soup",
        instructions="Chop carrots and potatoes. Simmer in stock.",
        category="Vegetarian",
        area="French",
        ingredients=(ExternalIngredient("Carrots", "2"), ExternalIngredient("Potatoes", "3")),
    ),
]


@pytest.fixture
async def knowledge_base(db_session: AsyncSession, fake_embedder: FakeEmbedder) -> None:
    """Import and index a few distinct recipes."""
    for external in KNOWLEDGE_BASE:
        recipe, _ = await import_recipe(db_session, external)
        await index_recipe(db_session, recipe, fake_embedder)


async def test_search_rejects_empty_query(client: AsyncClient) -> None:
    """An empty search query fails validation."""
    response = await client.post("/api/recipes/search", json={"query": ""})

    assert response.status_code == 422


async def test_search_returns_503_when_embedder_is_down(
    client: AsyncClient, fake_embedder: FakeEmbedder
) -> None:
    """Search reports the unavailable embedding service as 503."""
    fake_embedder.fail = True

    response = await client.post("/api/recipes/search", json={"query": "soup"})

    assert response.status_code == 503
    assert "embedder down" in response.json()["detail"]


async def test_ask_returns_503_when_embedder_is_down(
    client: AsyncClient, fake_embedder: FakeEmbedder
) -> None:
    """Ask reports the unavailable embedding service as 503."""
    fake_embedder.fail = True

    response = await client.post("/api/rag/ask", json={"question": "What's for dinner?"})

    assert response.status_code == 503


async def test_status_reports_index_and_models(client: AsyncClient) -> None:
    """Status counts recipes/chunks and reports a ready LLM when both models are pulled."""
    await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})

    body = (await client.get("/api/rag/status")).json()

    assert body == {
        "total_recipes": 1,
        "indexed_recipes": 1,
        "chunks": 2,
        "embedding_model": settings.ollama_embedding_model,
        "chat_model": settings.ollama_chat_model,
        "llm_available": True,
        "missing_models": [],
        "detail": None,
    }


async def test_status_reports_missing_models(client: AsyncClient, ollama_models: list[str]) -> None:
    """Models not pulled on the Ollama server make the LLM unavailable."""
    ollama_models.clear()

    body = (await client.get("/api/rag/status")).json()

    assert body["llm_available"] is False
    assert body["missing_models"] == [settings.ollama_embedding_model, settings.ollama_chat_model]
    assert "not pulled" in body["detail"]


@pytest.mark.postgres
@pytest.mark.usefixtures("knowledge_base")
async def test_search_ranks_relevant_recipe_first(client: AsyncClient) -> None:
    """The recipe sharing the query's words ranks first, once per recipe."""
    response = await client.post(
        "/api/recipes/search", json={"query": "chicken with rice", "limit": 3}
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["title"] == "Chicken Curry"
    assert len({hit["recipe_id"] for hit in results}) == len(results)
    assert results == sorted(results, key=lambda hit: hit["score"], reverse=True)
    assert "Chicken Curry" in results[0]["snippet"]


@pytest.mark.postgres
@pytest.mark.usefixtures("knowledge_base")
async def test_search_filters_by_category_case_insensitively(client: AsyncClient) -> None:
    """Only recipes of the requested category are returned."""
    response = await client.post(
        "/api/recipes/search", json={"query": "chicken", "category": "dessert"}
    )

    assert [hit["title"] for hit in response.json()["results"]] == ["Chocolate Cake"]


@pytest.mark.postgres
@pytest.mark.usefixtures("knowledge_base")
async def test_ask_grounds_answer_on_retrieved_recipes(
    client: AsyncClient, fake_chat_model: FakeChatModel
) -> None:
    """The chat model gets the retrieved recipes; sources are returned with the answer."""
    response = await client.post(
        "/api/rag/ask", json={"question": "What can I cook with chicken and rice?", "top_k": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == fake_chat_model.reply
    assert body["model"] == "fake-chat"
    assert [source["title"] for source in body["sources"]] == ["Chicken Curry"]
    prompt = fake_chat_model.received[0][1].content
    assert "Ingredients: chicken (500g), rice (1 cup)" in prompt
    assert "Chocolate Cake" not in prompt


@pytest.mark.postgres
async def test_ask_with_empty_knowledge_base_skips_llm(
    client: AsyncClient, fake_chat_model: FakeChatModel
) -> None:
    """Without indexed recipes the fixed answer is returned and the LLM is not called."""
    response = await client.post("/api/rag/ask", json={"question": "Anything?"})

    assert response.status_code == 200
    assert response.json()["model"] is None
    assert fake_chat_model.received == []


@pytest.mark.postgres
async def test_ask_returns_503_when_chat_model_is_down(
    client: AsyncClient, fake_chat_model: FakeChatModel
) -> None:
    """A failing chat model is reported as 503."""
    await client.post("/api/recipes", json={"title": "Soup", "instructions": "Simmer."})
    fake_chat_model.fail = True

    response = await client.post("/api/rag/ask", json={"question": "Soup?"})

    assert response.status_code == 503
