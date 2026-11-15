"""Tests for the TheMealDB client and its parsing helpers."""

from typing import Any

import httpx
import pytest
from app.clients.themealdb import TheMealDBClient, parse_meal, parse_measure
from app.core.exceptions import RecipeSourceError


def make_meal(
    meal_id: str = "52772", title: str = "Teriyaki Chicken", **extra: Any
) -> dict[str, Any]:
    """Build a TheMealDB `meal` payload with sensible defaults."""
    meal: dict[str, Any] = {
        "idMeal": meal_id,
        "strMeal": title,
        "strCategory": "Chicken",
        "strArea": "Japanese",
        "strInstructions": "Preheat oven.\r\nBake chicken.",
        "strTags": "Meat,Casserole, ",
        "strMealThumb": "https://img/x.jpg",
        "strSource": "",
        "strIngredient1": "soy sauce",
        "strMeasure1": "3/4 cup",
        "strIngredient2": "Chicken Breasts",
        "strMeasure2": "2",
        "strIngredient3": "",
        "strMeasure3": "",
        "strIngredient4": None,
        "strMeasure4": None,
    }
    meal.update(extra)
    return meal


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("3/4 cup", (0.75, "cup")),
        ("1 1/2 tsp", (1.5, "tsp")),
        ("1½ tbs", (1.5, "tbs")),
        ("½ cup", (0.5, "cup")),
        ("200g", (200.0, "g")),
        ("0,5 l", (0.5, "l")),
        ("2", (2.0, None)),
        ("Pinch", (None, "Pinch")),
        ("2-3 tbsp", (None, "2-3 tbsp")),
        ("0 g", (None, "g")),
        ("  ", (None, None)),
        (None, (None, None)),
        ("1 " + "very long unit " * 5, (1.0, "very long unit very long unit ve")),
    ],
)
def test_parse_measure(measure: str | None, expected: tuple[float | None, str | None]) -> None:
    """Free-text measures are split into a quantity and a (capped) unit."""
    assert parse_measure(measure) == expected


def test_parse_meal_maps_fields() -> None:
    """A meal becomes an `ExternalRecipe` with cleaned text, tags and ingredients."""
    recipe = parse_meal(make_meal())

    assert recipe is not None
    assert (recipe.source, recipe.external_id, recipe.title) == (
        "themealdb",
        "52772",
        "Teriyaki Chicken",
    )
    assert recipe.instructions == "Preheat oven.\nBake chicken."
    assert recipe.tags == ["Meat", "Casserole"]
    assert recipe.source_url is None
    assert [(i.name, i.quantity, i.unit) for i in recipe.ingredients] == [
        ("soy sauce", 0.75, "cup"),
        ("Chicken Breasts", 2.0, None),
    ]


def test_parse_meal_skips_incomplete_meals() -> None:
    """Meals without instructions can't be indexed and are dropped."""
    assert parse_meal(make_meal(strInstructions="  ")) is None


async def test_fetch_all_deduplicates_and_sorts() -> None:
    """Results of the per-letter searches are merged by id; empty letters are fine."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Serve two letters, one meal appearing under both."""
        letter = request.url.params["f"]
        meals = {
            "a": [make_meal("2", "Apple Pie"), make_meal("1", "Arrabiata")],
            "b": [make_meal("1", "Arrabiata")],
        }.get(letter)
        return httpx.Response(200, json={"meals": meals})

    async with TheMealDBClient("http://mealdb", transport=httpx.MockTransport(handler)) as client:
        recipes = await client.fetch_all("abc")

    assert [recipe.external_id for recipe in recipes] == ["1", "2"]


async def test_fetch_errors_raise_source_error() -> None:
    """HTTP failures surface as `RecipeSourceError`."""
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with TheMealDBClient("http://mealdb", transport=transport) as client:
        with pytest.raises(RecipeSourceError):
            await client.search_by_first_letter("a")
