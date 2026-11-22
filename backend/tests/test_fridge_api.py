"""Tests for the fridge items, diet profile and fridge suggestion endpoints."""

import json
from datetime import date, timedelta

from app.core.exceptions import LLMUnavailableError
from app.services.fridge import EMPTY_FRIDGE_NOTE, NO_MATCH_NOTE, REPLY_SCHEMA
from httpx import AsyncClient

from tests.fakes import FakeChatModel


async def create_recipe(client: AsyncClient, title: str, *ingredients: str) -> int:
    """Create (and index) a recipe from ingredient names; returns its id."""
    existing = {i["name"]: i["id"] for i in (await client.get("/api/ingredients")).json()}
    ids = []
    for name in ingredients:
        if name not in existing:
            response = await client.post("/api/ingredients", json={"name": name})
            existing[name] = response.json()["id"]
        ids.append(existing[name])
    response = await client.post(
        "/api/recipes",
        json={
            "title": title,
            "instructions": f"Cook the {', '.join(ingredients)}.",
            "ingredients": [{"ingredient_id": i} for i in ids],
        },
    )
    assert response.status_code == 201
    return int(response.json()["id"])


def llm_reply(*picks: dict[str, object]) -> str:
    """Serialize picks the way the LLM is asked to answer."""
    return json.dumps({"suggestions": list(picks)})


# --- Fridge items ----------------------------------------------------------------------


async def test_fridge_item_lifecycle(client: AsyncClient) -> None:
    """Items can be added (creating the ingredient), listed, updated and removed."""
    soon = (date.today() + timedelta(days=1)).isoformat()
    created = await client.post(
        "/api/fridge/items", json={"name": " Chicken Thighs ", "quantity": 500, "unit": "g"}
    )
    assert created.status_code == 201
    item = created.json()
    assert item["ingredient"]["name"] == "chicken thighs"

    await client.post("/api/fridge/items", json={"name": "spinach", "expires_on": soon})
    listed = (await client.get("/api/fridge/items")).json()
    assert [i["ingredient"]["name"] for i in listed] == ["spinach", "chicken thighs"]

    updated = await client.patch(f"/api/fridge/items/{item['id']}", json={"quantity": 250})
    assert updated.json()["quantity"] == 250
    assert updated.json()["unit"] == "g"

    assert (await client.delete(f"/api/fridge/items/{item['id']}")).status_code == 204
    assert (await client.delete(f"/api/fridge/items/{item['id']}")).status_code == 404
    assert len((await client.get("/api/fridge/items")).json()) == 1
    names = [i["name"] for i in (await client.get("/api/ingredients")).json()]
    assert "chicken thighs" in names


async def test_duplicate_fridge_item_conflicts(client: AsyncClient) -> None:
    """The same ingredient can't be added twice."""
    assert (await client.post("/api/fridge/items", json={"name": "rice"})).status_code == 201
    response = await client.post("/api/fridge/items", json={"name": "Rice"})
    assert response.status_code == 409
    assert "already in the fridge" in response.json()["detail"]


async def test_fridge_item_validation(client: AsyncClient) -> None:
    """Blank names and non-positive quantities are rejected."""
    assert (await client.post("/api/fridge/items", json={"name": "  "})).status_code == 422
    response = await client.post("/api/fridge/items", json={"name": "egg", "quantity": 0})
    assert response.status_code == 422
    assert (await client.patch("/api/fridge/items/99", json={})).status_code == 404


# --- Profile -------------------------------------------------------------------------------


async def test_profile_roundtrip(client: AsyncClient) -> None:
    """The profile starts empty, can be replaced and is normalized."""
    assert (await client.get("/api/profile")).json() == {
        "diets": [],
        "allergens": [],
        "excluded_ingredients": [],
    }
    response = await client.put(
        "/api/profile",
        json={
            "diets": ["vegan", "vegan"],
            "allergens": ["peanuts"],
            "excluded_ingredients": [" Coriander ", ""],
        },
    )
    assert response.status_code == 200
    expected = {"diets": ["vegan"], "allergens": ["peanuts"], "excluded_ingredients": ["coriander"]}
    assert response.json() == expected
    assert (await client.get("/api/profile")).json() == expected


async def test_profile_rejects_unknown_values(client: AsyncClient) -> None:
    """Only supported diets and allergens are accepted."""
    response = await client.put("/api/profile", json={"diets": ["paleo"]})
    assert response.status_code == 422


async def test_profile_options(client: AsyncClient) -> None:
    """The options list every diet with what it forbids, and the 14 allergens."""
    options = (await client.get("/api/profile/options")).json()
    vegan = next(d for d in options["diets"] if d["value"] == "vegan")
    assert {"milk", "eggs", "honey", "meat"} <= set(vegan["forbids"])
    assert len(options["allergens"]) == 14


# --- Suggestions ---------------------------------------------------------------------------


async def stock_fridge(client: AsyncClient) -> None:
    """Three recipes and a fridge that fits two of them."""
    await create_recipe(client, "Chicken Rice", "chicken", "rice", "butter", "salt")
    await create_recipe(client, "Spinach Rice", "rice", "spinach", "garlic")
    await create_recipe(client, "Lemon Cake", "flour", "sugar", "lemon", "eggs")
    soon = (date.today() + timedelta(days=1)).isoformat()
    for item in (
        {"name": "chicken thighs"},
        {"name": "rice"},
        {"name": "spinach", "expires_on": soon},
    ):
        await client.post("/api/fridge/items", json=item)


async def test_suggestions_without_llm(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """The rule-based ranking needs no LLM and reports fridge coverage."""
    await stock_fridge(client)

    response = await client.post("/api/fridge/suggestions", json={"use_llm": False})

    assert response.status_code == 200
    body = response.json()
    assert body["llm_used"] is False
    assert chat_model.received == []
    titles = [s["title"] for s in body["suggestions"]]
    assert titles == ["Spinach Rice", "Chicken Rice"]
    spinach = body["suggestions"][0]
    assert spinach["used_ingredients"] == ["rice", "spinach"]
    assert spinach["missing_ingredients"] == ["garlic"]
    assert spinach["expiring_used"] == ["spinach"]
    chicken = body["suggestions"][1]
    assert chicken["missing_ingredients"] == ["butter"]
    assert chicken["reason"] is None


async def test_suggestions_enforce_the_stored_profile(client: AsyncClient) -> None:
    """With a vegan profile chicken and butter are swapped by the rules."""
    await stock_fridge(client)
    await client.put("/api/profile", json={"diets": ["vegan"]})

    body = (await client.post("/api/fridge/suggestions", json={"use_llm": False})).json()

    chicken = next(s for s in body["suggestions"] if s["title"] == "Chicken Rice")
    swaps = {s["original"]: s["replacement"] for s in chicken["substitutions"]}
    assert swaps == {"chicken": "tofu", "butter": "olive oil"}
    assert all(s["source"] == "rules" for s in chicken["substitutions"])
    assert "chicken" not in chicken["used_ingredients"]


async def test_llm_ranks_and_its_output_is_validated(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """The LLM order wins, invented recipes and unsafe swaps are dropped."""
    await stock_fridge(client)
    chat_model.reply = llm_reply(
        {"recipe": 9, "reason": "Invented.", "substitutions": []},
        {
            "recipe": 2,
            "reason": "Uses your chicken.",
            "substitutions": [
                {"original": "butter", "replacement": "ghee"},
                {"original": "chicken", "replacement": "chickpeas"},
            ],
        },
    )

    body = (
        await client.post(
            "/api/fridge/suggestions",
            json={"limit": 2, "profile": {"diets": ["vegan"]}, "query": "something hearty"},
        )
    ).json()

    assert body["llm_used"] is True
    assert body["model"] == "fake-chat"
    assert chat_model.schemas == [REPLY_SCHEMA]
    first, second = body["suggestions"]
    assert first["title"] == "Chicken Rice"
    assert first["reason"] == "Uses your chicken."
    swaps = {s["original"]: (s["replacement"], s["source"]) for s in first["substitutions"]}
    assert swaps == {"chicken": ("chickpeas", "llm"), "butter": ("olive oil", "rules")}
    assert second["title"] == "Spinach Rice"
    assert second["reason"] is None
    assert any("unknown recipe [9]" in note for note in body["notes"])
    assert any("butter -> ghee: vegan: milk" in note for note in body["notes"])

    prompt = chat_model.received[0][1].content
    assert "Dietary constraints: diets: Vegan" in prompt
    assert "Wish: something hearty" in prompt
    assert "spinach (use soon)" in prompt
    assert "butter: FORBIDDEN (vegan: milk); allowed swaps: olive oil, vegan butter" in prompt


async def test_unparseable_llm_reply_falls_back(
    client: AsyncClient, chat_model: FakeChatModel
) -> None:
    """Invalid JSON yields the rule-based ranking with a note."""
    await stock_fridge(client)
    chat_model.reply = "Sure! I recommend the chicken."

    body = (await client.post("/api/fridge/suggestions", json={})).json()

    assert body["llm_used"] is False
    assert [s["title"] for s in body["suggestions"]] == ["Spinach Rice", "Chicken Rice"]
    assert "couldn't be parsed" in body["notes"][0]


async def test_fenced_llm_reply_is_accepted(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """A JSON answer wrapped in a Markdown fence is still parsed."""
    await stock_fridge(client)
    chat_model.reply = (
        "```json\n" + llm_reply({"recipe": 2, "reason": "Yum.", "substitutions": []}) + "\n```"
    )

    body = (await client.post("/api/fridge/suggestions", json={"limit": 1})).json()

    assert body["llm_used"] is True
    assert [s["title"] for s in body["suggestions"]] == ["Chicken Rice"]


async def test_llm_outage_falls_back(client: AsyncClient, chat_model: FakeChatModel) -> None:
    """An unreachable LLM doesn't fail the request."""
    await stock_fridge(client)
    chat_model.error = LLMUnavailableError("Ollama is unreachable")

    response = await client.post("/api/fridge/suggestions", json={})

    assert response.status_code == 200
    body = response.json()
    assert body["llm_used"] is False
    assert len(body["suggestions"]) == 2
    assert "AI ranking unavailable (Ollama is unreachable)" in body["notes"][0]


async def test_ingredient_override_and_expired_items(client: AsyncClient) -> None:
    """Request ingredients replace the fridge; expired stored items are skipped."""
    await stock_fridge(client)
    past = (date.today() - timedelta(days=2)).isoformat()
    await client.post("/api/fridge/items", json={"name": "lemon", "expires_on": past})

    stored = (await client.post("/api/fridge/suggestions", json={"use_llm": False})).json()
    assert "Skipped expired items: lemon." in stored["notes"]

    override = (
        await client.post(
            "/api/fridge/suggestions",
            json={"use_llm": False, "ingredients": ["Lemon", "eggs", "flour"], "max_missing": 1},
        )
    ).json()
    assert [s["title"] for s in override["suggestions"]] == ["Lemon Cake"]
    assert override["suggestions"][0]["missing_ingredients"] == ["sugar"]


async def test_empty_fridge_and_no_match(client: AsyncClient) -> None:
    """Empty fridges and impossible constraints are explained instead of failing."""
    empty = (await client.post("/api/fridge/suggestions", json={})).json()
    assert empty["suggestions"] == []
    assert empty["notes"] == [EMPTY_FRIDGE_NOTE]

    await create_recipe(client, "Black Pudding Fry", "black pudding", "onion")
    body = (
        await client.post(
            "/api/fridge/suggestions",
            json={"ingredients": ["onion"], "profile": {"diets": ["vegetarian"]}},
        )
    ).json()
    assert body["suggestions"] == []
    assert body["notes"] == [NO_MATCH_NOTE]
