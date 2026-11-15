"""Tests for the TheMealDB parser and client."""

from typing import Any

import httpx
import pytest
from app.core.exceptions import RecipeSourceError
from app.sources import ExternalIngredient, TheMealDBClient
from app.sources.themealdb import SOURCE_NAME, clean_instructions, parse_meal


def make_meal(**overrides: Any) -> dict[str, Any]:
    """Return a TheMealDB meal object with all 20 ingredient slots, like the real API."""
    meal: dict[str, Any] = {
        "idMeal": "52772",
        "strMeal": "Teriyaki Chicken Casserole",
        "strCategory": "Chicken",
        "strArea": "Japanese",
        "strCountry": "Japan",
        "strInstructions": "STEP 1\r\nPreheat oven.\r\n\r\nstep 2\r\nBake it.",
        "strMealThumb": "https://www.themealdb.com/images/media/meals/x.jpg",
        "strTags": "Meat, Casserole,,Meat",
        "strSource": "",
    }
    for i in range(1, 21):
        meal[f"strIngredient{i}"] = ""
        meal[f"strMeasure{i}"] = ""
    meal.update(strIngredient1="Soy Sauce", strMeasure1="3/4 cup")
    meal.update(strIngredient2=" Water ", strMeasure2=" ")
    meal.update(strIngredient3=None, strMeasure3=None)
    meal.update(overrides)
    return meal


def test_parse_meal_maps_fields() -> None:
    """Fields are mapped, blanks dropped, tags deduplicated and instructions cleaned."""
    recipe = parse_meal(make_meal())

    assert recipe is not None
    assert recipe.source == SOURCE_NAME
    assert recipe.external_id == "52772"
    assert recipe.title == "Teriyaki Chicken Casserole"
    assert recipe.category == "Chicken"
    assert recipe.area == "Japanese"
    assert recipe.tags == ("Meat", "Casserole")
    assert recipe.source_url is None
    assert recipe.instructions == "Preheat oven.\n\nBake it."
    assert recipe.ingredients == (
        ExternalIngredient("Soy Sauce", "3/4 cup"),
        ExternalIngredient("Water", None),
    )


@pytest.mark.parametrize(
    ("area", "country", "expected_area"),
    [
        ("Italian", "Italy", "Italian"),
        ("", "France", "French"),
        ("France", "France", "French"),
        (None, "Atlantis", "Atlantis"),
    ],
)
def test_parse_meal_normalizes_area_to_cuisine(
    area: str | None, country: str, expected_area: str
) -> None:
    """Country names (in either field) become the cuisine; unknown names are kept."""
    recipe = parse_meal(make_meal(strArea=area, strCountry=country), {"France": "French"})

    assert recipe is not None
    assert recipe.area == expected_area


@pytest.mark.parametrize("missing", ["idMeal", "strMeal", "strInstructions"])
def test_parse_meal_rejects_incomplete_meals(missing: str) -> None:
    """A meal without id, name or instructions is skipped."""
    assert parse_meal(make_meal(**{missing: " "})) is None


def test_clean_instructions_collapses_lines_into_paragraphs() -> None:
    """Single line breaks join a paragraph; blank lines and step headings separate them."""
    assert clean_instructions("Mix.\nStir.\r\n\r\n\r\nSTEP 3:\nServe.") == "Mix. Stir.\n\nServe."


API_PREFIX = "/api/json/v1/1"
AREAS = {"meals": [{"strArea": "Japanese", "strCountry": "Japan"}, {"strArea": None}]}


def _client(handler: Any) -> TheMealDBClient:
    """Build a client whose HTTP calls are answered by `handler`."""
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=f"https://mealdb.test{API_PREFIX}"
    )
    return TheMealDBClient(http)


async def test_area_by_country_skips_incomplete_entries() -> None:
    """The area list becomes a country -> cuisine mapping."""
    client = _client(lambda request: httpx.Response(200, json=AREAS))

    assert await client.area_by_country() == {"Japan": "Japanese"}


async def test_iter_meals_queries_each_letter() -> None:
    """Every letter is requested from the search endpoint; `null` results are fine."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the area list, one meal for "a" and none for "b"."""
        if request.url.path == f"{API_PREFIX}/list.php":
            return httpx.Response(200, json=AREAS)
        assert request.url.path == f"{API_PREFIX}/search.php"
        letter = request.url.params["f"]
        requested.append(letter)
        meal = make_meal(idMeal="1", strMeal="Apple Pie", strArea=None, strCountry="Japan")
        return httpx.Response(200, json={"meals": [meal] if letter == "a" else None})

    meals = [meal async for meal in _client(handler).iter_meals("ab")]

    assert requested == ["a", "b"]
    assert [(meal.title, meal.area) for meal in meals] == [("Apple Pie", "Japanese")]


async def test_meal_by_id() -> None:
    """A meal is looked up by id; unknown ids give `None`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve the area list and know only meal 52772."""
        if request.url.path == f"{API_PREFIX}/list.php":
            return httpx.Response(200, json=AREAS)
        found = request.url.params["i"] == "52772"
        return httpx.Response(200, json={"meals": [make_meal()] if found else None})

    client = _client(handler)

    found = await client.meal_by_id("52772")
    assert found is not None
    assert found.external_id == "52772"
    assert await client.meal_by_id("1") is None


async def test_http_errors_become_recipe_source_errors() -> None:
    """Server errors are reported as `RecipeSourceError`."""
    client = _client(lambda request: httpx.Response(500))

    with pytest.raises(RecipeSourceError):
        await client.meals_by_first_letter("a")


async def test_meals_by_first_letter_validates_letter() -> None:
    """Only single ASCII letters are accepted."""
    client = _client(lambda request: httpx.Response(200, json={"meals": None}))

    with pytest.raises(ValueError, match="single ASCII letter"):
        await client.meals_by_first_letter("ab")
