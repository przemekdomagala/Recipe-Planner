"""Client and parser for TheMealDB (https://www.themealdb.com/api.php)."""

import re
import string
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import RecipeSourceError
from app.sources.base import ExternalIngredient, ExternalRecipe

SOURCE_NAME = "themealdb"
"""Value of `Recipe.source` for recipes imported from TheMealDB."""

# TheMealDB stores ingredients in 20 numbered column pairs (strIngredient1/strMeasure1, ...).
_MAX_INGREDIENTS = 20
# Lines such as "STEP 1" or "step 2:" only number the paragraphs that follow them.
_STEP_HEADING = re.compile(r"^\s*step\s*\d+\s*[:.]?\s*$", re.IGNORECASE)


def _clean(value: object) -> str | None:
    """Return `value` stripped, or `None` if it is not a non-blank string."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def clean_instructions(text: str) -> str:
    """Normalize line endings, drop "step N" headings and collapse runs of blank lines.

    Args:
        text: Raw `strInstructions` value.

    Returns:
        Paragraphs separated by a single blank line.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or _STEP_HEADING.match(line):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def parse_meal(
    meal: Mapping[str, Any], area_by_country: Mapping[str, str] | None = None
) -> ExternalRecipe | None:
    """Map one TheMealDB meal object onto an `ExternalRecipe`.

    The area should be a cuisine ("French"), but newer meals leave `strArea` empty and
    only set `strCountry`, and some put a country into `strArea` ("France").
    `area_by_country` maps such country names to the cuisine so that filtering by area
    stays consistent; unknown names are kept as they are.

    Args:
        meal: A meal as returned by the TheMealDB JSON API.
        area_by_country: Cuisine name per country, see `TheMealDBClient.area_by_country`.

    Returns:
        The parsed recipe, or `None` if the meal lacks an id, a name or instructions.
    """
    external_id = _clean(meal.get("idMeal"))
    title = _clean(meal.get("strMeal"))
    raw_instructions = _clean(meal.get("strInstructions"))
    if not (external_id and title and raw_instructions):
        return None
    instructions = clean_instructions(raw_instructions)

    ingredients = tuple(
        ExternalIngredient(name=name, measure=_clean(meal.get(f"strMeasure{i}")))
        for i in range(1, _MAX_INGREDIENTS + 1)
        if (name := _clean(meal.get(f"strIngredient{i}")))
    )
    raw_tags = (_clean(meal.get("strTags")) or "").split(",")
    tags = tuple(dict.fromkeys(tag.strip() for tag in raw_tags if tag.strip()))
    area = _clean(meal.get("strArea")) or _clean(meal.get("strCountry"))
    if area is not None:
        area = (area_by_country or {}).get(area, area)

    return ExternalRecipe(
        source=SOURCE_NAME,
        external_id=external_id,
        title=title,
        instructions=instructions,
        category=_clean(meal.get("strCategory")),
        area=area,
        tags=tags,
        image_url=_clean(meal.get("strMealThumb")),
        source_url=_clean(meal.get("strSource")),
        ingredients=ingredients,
    )


class TheMealDBClient:
    """Read-only async client for the TheMealDB JSON API."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        """Wrap an HTTP client.

        Args:
            http: HTTP client whose `base_url` is the API root including the key, e.g.
                "https://www.themealdb.com/api/json/v1/1".
        """
        self._http = http

    async def area_by_country(self) -> dict[str, str]:
        """Return the cuisine name of every country TheMealDB knows, e.g. "France" -> "French".

        Raises:
            RecipeSourceError: If the API cannot be reached or replies with garbage.
        """
        body = await self._get("/list.php", {"a": "list"})
        return {
            country: area
            for item in body.get("meals") or []
            if (country := _clean(item.get("strCountry"))) and (area := _clean(item.get("strArea")))
        }

    async def meals_by_first_letter(
        self, letter: str, area_by_country: Mapping[str, str] | None = None
    ) -> list[ExternalRecipe]:
        """Return all meals whose name starts with `letter`.

        Args:
            letter: A single ASCII letter.
            area_by_country: Passed on to `parse_meal`.

        Raises:
            ValueError: If `letter` is not a single ASCII letter.
            RecipeSourceError: If the API cannot be reached or replies with garbage.
        """
        if len(letter) != 1 or letter.lower() not in string.ascii_lowercase:
            raise ValueError(f"Expected a single ASCII letter, got {letter!r}")
        body = await self._get("/search.php", {"f": letter.lower()})
        return self._parse_meals(body, area_by_country)

    async def meal_by_id(self, meal_id: str) -> ExternalRecipe | None:
        """Return one meal by its TheMealDB id, or `None` if it does not exist.

        Raises:
            RecipeSourceError: If the API cannot be reached or replies with garbage.
        """
        body = await self._get("/lookup.php", {"i": meal_id})
        meals = self._parse_meals(body, await self.area_by_country())
        return meals[0] if meals else None

    async def iter_meals(
        self, letters: Iterable[str] = string.ascii_lowercase
    ) -> AsyncIterator[ExternalRecipe]:
        """Yield every meal whose name starts with one of `letters`, letter by letter.

        Iterating over the whole alphabet (the default) walks the entire database, since
        the free API has no "list everything" endpoint.

        Raises:
            RecipeSourceError: If the API cannot be reached or replies with garbage.
        """
        area_by_country = await self.area_by_country()
        for letter in letters:
            for meal in await self.meals_by_first_letter(letter, area_by_country):
                yield meal

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        """GET a JSON object from the API, translating transport and decoding errors."""
        try:
            response = await self._http.get(path, params=params)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RecipeSourceError(f"TheMealDB request {path} failed: {exc}") from exc
        if not isinstance(body, dict):
            raise RecipeSourceError(f"TheMealDB request {path} returned unexpected JSON")
        return body

    @staticmethod
    def _parse_meals(
        body: Mapping[str, Any], area_by_country: Mapping[str, str] | None
    ) -> list[ExternalRecipe]:
        """Parse the `meals` array of a reply; the API returns `null` when nothing matches."""
        meals = body.get("meals") or []
        return [
            recipe for meal in meals if (recipe := parse_meal(meal, area_by_country)) is not None
        ]


@asynccontextmanager
async def themealdb_client(settings: Settings) -> AsyncIterator[TheMealDBClient]:
    """Yield a `TheMealDBClient` backed by an HTTP client that is closed on exit.

    Args:
        settings: Application settings (API base URL and timeout).
    """
    async with httpx.AsyncClient(
        base_url=settings.themealdb_base_url, timeout=settings.themealdb_timeout_seconds
    ) as http:
        yield TheMealDBClient(http)
