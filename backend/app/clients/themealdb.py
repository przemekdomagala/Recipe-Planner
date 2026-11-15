"""Client for TheMealDB (https://www.themealdb.com/api.php), the seed recipe dataset.

The free API has no "list everything" endpoint, so the full catalogue is fetched with one
`search.php?f=<first letter>` request per letter/digit.
"""

import asyncio
import logging
import re
import string
from collections.abc import Iterable
from types import TracebackType
from typing import Any, Self

import httpx

from app.core.config import settings
from app.core.exceptions import RecipeSourceError
from app.schemas.ingest import ExternalIngredient, ExternalRecipe

logger = logging.getLogger(__name__)

SOURCE = "themealdb"
ALL_LETTERS = string.ascii_lowercase + string.digits
_MAX_INGREDIENTS = 20  # TheMealDB stores ingredients in strIngredient1..20.
_MAX_UNIT_LENGTH = 32

_UNICODE_FRACTIONS = {
    "½": "1/2",
    "⅓": "1/3",
    "⅔": "2/3",
    "¼": "1/4",
    "¾": "3/4",
    "⅕": "1/5",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}
_NUMBER = r"\d+(?:[.,]\d+)?"
# "1 1/2", "3/4", "200", "0.5" at the start of a measure, followed by the unit.
_MEASURE_RE = re.compile(
    rf"^(?P<qty>\d+\s+\d+/\d+|\d+/\d+|{_NUMBER})(?!\s*[-\u2013]\s*\d)\s*(?P<unit>.*)$"
)


def parse_measure(measure: str | None) -> tuple[float | None, str | None]:
    """Split a free-text measure such as `"1 1/2 cups"` into `(1.5, "cups")`.

    Text without a leading number (`"Pinch"`, `"to taste"`) and ranges (`"2-3 tbsp"`) are
    kept whole as the unit, with no quantity.

    Args:
        measure: Raw measure text from the source.

    Returns:
        `(quantity, unit)`; either may be `None`. The unit is capped at 32 characters.
    """
    text = " ".join((measure or "").split())
    for symbol, fraction in _UNICODE_FRACTIONS.items():
        text = re.sub(rf"(\d)\s*{symbol}", rf"\1 {fraction}", text).replace(symbol, fraction)
    text = text.strip()
    if not text:
        return None, None

    quantity: float | None = None
    unit = text
    match = _MEASURE_RE.match(text)
    if match:
        quantity = _to_number(match["qty"])
        unit = match["unit"].strip()
    if quantity is not None and quantity <= 0:
        quantity = None
    return quantity, (unit[:_MAX_UNIT_LENGTH].strip() or None)


def _to_number(text: str) -> float:
    """Convert `"1 1/2"`, `"3/4"` or `"0,5"` to a float."""
    total = 0.0
    for part in text.split():
        if "/" in part:
            numerator, denominator = part.split("/")
            total += float(numerator) / float(denominator) if float(denominator) else 0.0
        else:
            total += float(part.replace(",", "."))
    return round(total, 4)


def _clean(value: object) -> str | None:
    """Return a stripped string, or `None` for missing/blank values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_meal(meal: dict[str, Any]) -> ExternalRecipe | None:
    """Convert one TheMealDB `meal` object into an `ExternalRecipe`.

    Returns:
        The recipe, or `None` if it lacks an id, a title or instructions.
    """
    meal_id, title = _clean(meal.get("idMeal")), _clean(meal.get("strMeal"))
    instructions = _clean((meal.get("strInstructions") or "").replace("\r\n", "\n"))
    if not (meal_id and title and instructions):
        return None

    ingredients: list[ExternalIngredient] = []
    for i in range(1, _MAX_INGREDIENTS + 1):
        name = _clean(meal.get(f"strIngredient{i}"))
        if name is None:
            continue
        quantity, unit = parse_measure(_clean(meal.get(f"strMeasure{i}")))
        ingredients.append(ExternalIngredient(name=name[:120], quantity=quantity, unit=unit))

    tags = [tag.strip() for tag in (meal.get("strTags") or "").split(",") if tag.strip()]
    return ExternalRecipe(
        source=SOURCE,
        external_id=meal_id,
        title=title[:200],
        instructions=instructions,
        category=_clean(meal.get("strCategory")),
        area=_clean(meal.get("strArea")),
        tags=tags,
        thumbnail_url=_clean(meal.get("strMealThumb")),
        source_url=_clean(meal.get("strSource")),
        ingredients=ingredients,
    )


class TheMealDBClient:
    """Async client fetching recipes from TheMealDB; use as an async context manager."""

    def __init__(
        self,
        base_url: str = settings.themealdb_base_url,
        *,
        concurrency: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Configure the client.

        Args:
            base_url: API root including the key, e.g. `https://www.themealdb.com/api/json/v1/1`.
            concurrency: Maximum number of requests in flight.
            transport: Optional custom transport (used by tests).
        """
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=30.0, transport=transport
        )
        self._semaphore = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> Self:
        """Return the client itself."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def search_by_first_letter(self, letter: str) -> list[ExternalRecipe]:
        """Return all meals whose name starts with `letter`.

        Raises:
            RecipeSourceError: If the request fails or the payload is malformed.
        """
        async with self._semaphore:
            try:
                response = await self._http.get("/search.php", params={"f": letter})
                response.raise_for_status()
                meals = response.json().get("meals") or []
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                raise RecipeSourceError(f"TheMealDB request for '{letter}' failed: {exc}") from exc

        recipes = []
        for meal in meals:
            recipe = parse_meal(meal) if isinstance(meal, dict) else None
            if recipe is None:
                logger.warning("Skipping incomplete TheMealDB meal: %r", meal)
            else:
                recipes.append(recipe)
        return recipes

    async def fetch_all(self, letters: Iterable[str] = ALL_LETTERS) -> list[ExternalRecipe]:
        """Fetch every meal whose name starts with one of `letters`, de-duplicated by id."""
        batches = await asyncio.gather(*(self.search_by_first_letter(c) for c in letters))
        unique = {recipe.external_id: recipe for batch in batches for recipe in batch}
        return sorted(unique.values(), key=lambda recipe: int(recipe.external_id))
