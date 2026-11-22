"""Tests for fridge suggestions (`POST /api/fridge/suggestions`)."""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from app.core.config import settings
from app.core.exceptions import LLMUnavailableError
from app.schemas import ExternalIngredient, ExternalRecipe
from app.services.ingestion import ingest_recipes
from app.services.llm import ChatMessage
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fakes import FakeChatModel, FakeEmbedder

URL = "/api/fridge/suggestions"
FRIDGE = ["Chicken", "rice", "spinach", "garlic", "eggs"]

RECIPES: list[tuple[str, str, list[str]]] = [
    ("Chicken Fried Rice", "Chicken", ["chicken", "rice", "eggs", "spring onion", "oil"]),
    (
        "Creamy Garlic Chicken",
        "Chicken",
        ["chicken breast", "garlic", "butter", "double cream", "spinach", "salt"],
    ),
    ("Spinach Rice Pilaf", "Vegetarian", ["rice", "spinach", "onion", "olive oil"]),
    ("Prawn Linguine", "Seafood", ["prawns", "linguine", "garlic", "butter", "parmesan"]),
]

Reply = Callable[[Sequence[ChatMessage]], str]


@pytest.fixture(autouse=True)
async def knowledge_base(db_session: AsyncSession, embedder: FakeEmbedder) -> None:
    """Ingest a small knowledge base for every test in this module."""
    await ingest_recipes(
        db_session,
        [
            ExternalRecipe(
                source="test",
                external_id=str(index),
                title=title,
                category=category,
                instructions=f"Cook the {title.lower()}.",
                ingredients=[ExternalIngredient(name=name) for name in ingredients],
            )
            for index, (title, category, ingredients) in enumerate(RECIPES)
        ],
        embedder,
    )


def numbers(messages: Sequence[ChatMessage]) -> dict[str, int]:
    """Map candidate titles to their numbers in the prompt."""
    found = re.findall(r"^\[(\d+)\] (.+?)(?: \(.*\))?$", messages[-1].content, re.MULTILINE)
    return {title: int(number) for number, title in found}


def picks(*choices: tuple[str, dict[str, Any]]) -> Reply:
    """A model reply choosing the given titles (with extra fields), best first."""

    def reply(messages: Sequence[ChatMessage]) -> str:
        """Resolve the titles to prompt numbers and answer with JSON."""
        by_title = numbers(messages)
        suggestions = [{"recipe": by_title[title], **extra} for title, extra in choices]
        return json.dumps({"suggestions": suggestions})

    return reply


async def suggest(client: AsyncClient, **payload: Any) -> dict[str, Any]:
    """POST a fridge request and return the JSON body, asserting success."""
    response = await client.post(URL, json={"ingredients": FRIDGE, **payload})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_llm_ranking_is_returned_with_coverage(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """The model's order and reasons are used; coverage and missing items are computed."""
    chat_model.reply = picks(
        ("Spinach Rice Pilaf", {"reason": "Uses your rice and spinach."}),
        ("Chicken Fried Rice", {"reason": "Classic leftover dish."}),
    )

    body = await suggest(client, max_suggestions=2, note="quick dinner")

    assert body["model"] == "fake-chat"
    assert body["ingredients"] == ["chicken", "rice", "spinach", "garlic", "eggs"]
    first, second = body["suggestions"]
    assert (first["rank"], first["title"]) == (1, "Spinach Rice Pilaf")
    assert first["reason"] == "Uses your rice and spinach."
    assert first["used_ingredients"] == ["rice", "spinach", "olive oil"]
    assert first["missing_ingredients"] == ["onion"]
    assert first["coverage"] == pytest.approx(2 / 3, abs=1e-3)  # olive oil is a staple
    assert second["title"] == "Chicken Fried Rice"
    assert body["excluded"] == []

    (schema,) = chat_model.schemas
    assert schema is not None
    assert schema["properties"]["suggestions"]["maxItems"] == 2
    assert schema["$defs"]["_LLMPick"]["properties"]["recipe"]["enum"] == [1, 2, 3, 4]
    assert schema["$defs"]["_LLMPick"]["properties"]["reason"]["maxLength"] > 0
    system, user = chat_model.received[0]
    assert "candidates ONLY" in system.content
    assert user.content.startswith("Fridge: chicken, rice, spinach, garlic, eggs\n")
    assert "Restrictions: none\nRequest: quick dinner\nChoose up to 2 recipes." in user.content


async def test_unadapted_conflicts_are_excluded_and_slots_refilled(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """A pick that still contains an allergen is dropped; compliant recipes fill its slot."""
    chat_model.reply = picks(("Creamy Garlic Chicken", {"reason": "Rich and creamy."}))

    body = await suggest(client, allergens=["dairy"])

    titles = [s["title"] for s in body["suggestions"]]
    assert "Creamy Garlic Chicken" not in titles
    assert set(titles) == {"Chicken Fried Rice", "Spinach Rice Pilaf"}
    assert all(s["reason"] is None for s in body["suggestions"])  # deterministic top-up
    excluded = {e["title"]: e["reasons"] for e in body["excluded"]}
    assert excluded["Creamy Garlic Chicken"] == [
        "butter: contains dairy",
        "double cream: contains dairy",
    ]
    assert "Conflicts: butter: contains dairy; double cream: contains dairy" in (
        chat_model.received[0][1].content
    )


async def test_safe_adaptations_make_a_recipe_eligible(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """Replacing every conflicting ingredient with a safe substitute passes the check."""
    adaptations = [
        {"replace": "Butter", "substitute": "olive oil", "note": "Sear in oil."},
        {"replace": "cream", "substitute": "coconut cream", "note": "Same amount."},
        {"replace": None, "substitute": None, "note": "Add lemon at the end."},
        {"replace": "salt", "substitute": "spinach", "note": "Already in the recipe."},
    ]
    chat_model.reply = picks(
        ("Creamy Garlic Chicken", {"reason": "Dairy-free take.", "adaptations": adaptations})
    )

    body = await suggest(client, allergens=["dairy"], max_suggestions=1)

    (suggestion,) = body["suggestions"]
    assert suggestion["title"] == "Creamy Garlic Chicken"
    assert suggestion["adaptations"] == [
        {"replace": "butter", "substitute": "olive oil", "note": "Sear in oil."},
        {"replace": "double cream", "substitute": "coconut cream", "note": "Same amount."},
        {"replace": None, "substitute": None, "note": "Add lemon at the end."},
        {"replace": "salt", "substitute": "spinach", "note": "Already in the recipe."},
    ]
    assert suggestion["missing_ingredients"] == ["coconut cream"]
    assert "butter" not in suggestion["missing_ingredients"]
    assert "Creamy Garlic Chicken" not in {e["title"] for e in body["excluded"]}


async def test_unsafe_or_hallucinated_adaptations_are_rejected(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """An unsafe substitute or a swap of an absent ingredient doesn't clear a conflict."""
    adaptations = [
        {"replace": "butter", "substitute": "ghee", "note": "Nutty flavour."},
        {"replace": "double cream", "substitute": "oat milk", "note": "Lighter."},
        {"replace": "truffle", "substitute": "mushroom", "note": "Cheaper."},
        {"replace": "garlic", "substitute": "Garlic", "note": ""},
    ]
    chat_model.reply = picks(("Creamy Garlic Chicken", {"adaptations": adaptations}))

    body = await suggest(client, allergens=["dairy"])

    assert "Creamy Garlic Chicken" not in [s["title"] for s in body["suggestions"]]
    excluded = {e["title"]: e["reasons"] for e in body["excluded"]}
    assert excluded["Creamy Garlic Chicken"] == ["butter: contains dairy"]


async def test_invalid_picks_are_ignored(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """Unknown recipe numbers and duplicates are skipped."""

    def reply(messages: Sequence[ChatMessage]) -> str:
        """Pick a non-existent recipe and the pilaf twice."""
        pilaf = numbers(messages)["Spinach Rice Pilaf"]
        return json.dumps({"suggestions": [{"recipe": 99}, {"recipe": pilaf}, {"recipe": pilaf}]})

    chat_model.reply = reply

    body = await suggest(client, max_suggestions=2)

    assert body["suggestions"][0]["title"] == "Spinach Rice Pilaf"
    assert len({s["recipe_id"] for s in body["suggestions"]}) == 2


async def test_malformed_reply_falls_back_to_deterministic_ranking(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """Non-JSON output still yields compliant suggestions, without reasons or model."""
    chat_model.reply = "Sure! Here are some recipes: ..."

    body = await suggest(client, diet="vegetarian")

    assert body["model"] is None
    assert [s["title"] for s in body["suggestions"]] == ["Spinach Rice Pilaf"]
    assert {e["title"] for e in body["excluded"]} == {
        "Chicken Fried Rice",
        "Creamy Garlic Chicken",
        "Prawn Linguine",
    }


async def test_heavily_conflicting_recipes_stay_out_of_the_prompt(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """Recipes with more conflicts than `fridge_max_conflicts` aren't offered to the model."""
    chat_model.reply = picks()

    body = await suggest(client, diet="vegan")

    prompt = numbers(chat_model.received[0])
    assert "Creamy Garlic Chicken" not in prompt  # chicken, butter and cream
    assert "Spinach Rice Pilaf" in prompt
    reasons = {e["title"]: e["reasons"] for e in body["excluded"]}
    assert len(reasons["Creamy Garlic Chicken"]) > settings.fridge_max_conflicts
    assert [s["title"] for s in body["suggestions"]] == ["Spinach Rice Pilaf"]


async def test_avoided_ingredients_are_enforced(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """User-listed dislikes are treated like allergens."""
    chat_model.reply = picks(("Spinach Rice Pilaf", {}))

    body = await suggest(client, avoid=["Onion"])

    assert "Spinach Rice Pilaf" not in [s["title"] for s in body["suggestions"]]
    assert {e["title"]: e["reasons"] for e in body["excluded"]} == {
        "Spinach Rice Pilaf": ["onion: you avoid onion"],
        "Chicken Fried Rice": ["spring onion: you avoid onion"],
    }


async def test_no_candidates_skips_the_llm(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """When filters leave nothing to rank, the model isn't called."""
    body = await suggest(client, filters={"area": "Peruvian"})

    assert body["suggestions"] == []
    assert body["model"] is None
    assert chat_model.received == []


async def test_llm_unavailable_is_503(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """Transport failures are not masked by the fallback."""
    chat_model.error = LLMUnavailableError("Ollama is unreachable")

    response = await client.post(URL, json={"ingredients": FRIDGE})

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"ingredients": []},
        {"ingredients": ["  ", ""]},
        {"ingredients": ["rice"], "allergens": ["shellfish"]},
        {"ingredients": ["rice"], "diet": "keto"},
        {"ingredients": ["rice"], "max_suggestions": 6},
    ],
)
async def test_invalid_requests_are_rejected(client: AsyncClient, payload: dict[str, Any]) -> None:
    """Empty fridges and unknown allergens or diets fail validation."""
    response = await client.post(URL, json=payload)

    assert response.status_code == 422
