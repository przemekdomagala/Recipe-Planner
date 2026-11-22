"""Tests for the RAG assistant (`POST /api/assistant/ask`)."""

from app.core.exceptions import LLMModelNotFoundError
from app.services.assistant import NO_MATCH_ANSWER
from httpx import AsyncClient

from tests.fakes import FakeChatModel


async def _create_recipe(client: AsyncClient, title: str, instructions: str) -> None:
    """Create (and thereby index) a recipe with a single ingredient."""
    ingredient = await client.post("/api/ingredients", json={"name": f"{title} base"})
    response = await client.post(
        "/api/recipes",
        json={
            "title": title,
            "instructions": instructions,
            "area": "Japanese",
            "ingredients": [
                {"ingredient_id": ingredient.json()["id"], "quantity": 2, "unit": "cups"}
            ],
        },
    )
    assert response.status_code == 201


async def test_ask_grounds_the_llm_in_retrieved_recipes(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """The prompt carries the retrieved recipes; the response returns answer and sources."""
    await _create_recipe(client, "Teriyaki Chicken", "Glaze chicken with teriyaki sauce.")
    await _create_recipe(client, "Miso Soup", "Whisk miso into dashi.")

    response = await client.post(
        "/api/assistant/ask", json={"question": "How do I glaze teriyaki chicken?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == chat_model.reply
    assert body["model"] == "fake-chat"
    assert body["sources"][0]["title"] == "Teriyaki Chicken"

    system, user = chat_model.received[0]
    assert system.role == "system"
    assert "ONLY the recipes provided" in system.content
    assert user.content.startswith("Context recipes:\n\n[1] Teriyaki Chicken (Japanese)")
    assert "- 2 cups teriyaki chicken base" in user.content
    assert user.content.endswith("Question: How do I glaze teriyaki chicken?")


async def test_ask_without_matches_skips_the_llm(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """An empty knowledge base yields a fixed answer and no LLM call."""
    response = await client.post("/api/assistant/ask", json={"question": "Anything vegan?"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "Anything vegan?",
        "answer": NO_MATCH_ANSWER,
        "model": None,
        "sources": [],
    }
    assert chat_model.received == []


async def test_ask_reports_missing_model_as_503(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """LLM errors are mapped to HTTP 503 with an actionable message."""
    await _create_recipe(client, "Ramen", "Boil noodles.")
    chat_model.error = LLMModelNotFoundError("gemma4:31b")

    response = await client.post("/api/assistant/ask", json={"question": "ramen?"})

    assert response.status_code == 503
    assert "ollama pull gemma4:31b" in response.json()["detail"]
