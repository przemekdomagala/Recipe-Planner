"""Importing recipes from external sources into the database (and the vector store)."""

from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import Self

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.llm import Embedder
from app.models import Recipe, RecipeIngredient
from app.rag import index_recipe
from app.schemas import normalize_ingredient_name
from app.schemas.ingredient import INGREDIENT_NAME_MAX_LENGTH
from app.sources import ExternalIngredient, ExternalRecipe, parse_measure

_MEASURE_MAX_LENGTH = 200
_TITLE_MAX_LENGTH = 200


@dataclass(frozen=True, slots=True)
class IngredientLine:
    """An ingredient line after normalization and measure parsing."""

    name: str
    quantity: float | None
    unit: str | None
    measure: str | None

    def merge(self, other: Self) -> Self:
        """Combine two lines for the same ingredient into one.

        Quantities are summed when both are in the same unit; otherwise the parsed amount
        is dropped as ambiguous. The raw measures are always kept, joined with " + ".
        """
        if other.quantity is None and other.measure is None:
            return self
        if self.quantity is None and self.measure is None:
            return other
        if self.quantity is not None and other.quantity is not None and self.unit == other.unit:
            quantity, unit = self.quantity + other.quantity, self.unit
        else:
            quantity, unit = None, None
        measures = [m for m in (self.measure, other.measure) if m]
        measure = " + ".join(measures)[:_MEASURE_MAX_LENGTH] if measures else None
        return replace(self, quantity=quantity, unit=unit, measure=measure)


@dataclass(slots=True)
class ImportResult:
    """Counters reported by `import_recipes`."""

    created: int = 0
    updated: int = 0
    indexed: int = 0

    @property
    def imported(self) -> int:
        """Number of recipes created or updated."""
        return self.created + self.updated


def build_ingredient_lines(ingredients: Iterable[ExternalIngredient]) -> list[IngredientLine]:
    """Normalize names, parse measures and merge repeated ingredients, keeping first order.

    Sources sometimes list an ingredient twice (e.g. salt for the dough and the filling),
    but a recipe may link each ingredient only once.

    Args:
        ingredients: Ingredient lines as published by the source.
    """
    lines: dict[str, IngredientLine] = {}
    for item in ingredients:
        name = normalize_ingredient_name(item.name)[:INGREDIENT_NAME_MAX_LENGTH]
        if not name:
            continue
        parsed = parse_measure(item.measure)
        measure = item.measure[:_MEASURE_MAX_LENGTH] if item.measure else None
        line = IngredientLine(name, parsed.quantity, parsed.unit, measure)
        lines[name] = lines[name].merge(line) if name in lines else line
    return list(lines.values())


async def import_recipe(db: AsyncSession, external: ExternalRecipe) -> tuple[Recipe, bool]:
    """Create or update the recipe identified by `(external.source, external.external_id)`.

    Ingredients are created on demand. Ingredient links are updated in place, because
    deleting and re-inserting a link with the same primary key in one flush fails.

    Args:
        db: Database session; committed on success.
        external: Recipe fetched from the source.

    Returns:
        The persisted recipe (ingredients loaded) and whether it was newly created.
    """
    lines = build_ingredient_lines(external.ingredients)
    ingredients = await crud.ingredient.get_or_create_ingredients(db, (line.name for line in lines))

    recipe = await crud.recipe.get_recipe_by_external_id(db, external.source, external.external_id)
    created = recipe is None
    if recipe is None:
        recipe = Recipe(source=external.source, external_id=external.external_id, ingredients=[])
        db.add(recipe)
    recipe.title = external.title[:_TITLE_MAX_LENGTH]
    recipe.instructions = external.instructions
    recipe.category = external.category
    recipe.area = external.area
    recipe.tags = list(external.tags)
    recipe.image_url = external.image_url
    recipe.source_url = external.source_url

    links = {link.ingredient_id: link for link in recipe.ingredients}
    wanted_ids: set[int] = set()
    for position, line in enumerate(lines):
        ingredient = ingredients[line.name]
        wanted_ids.add(ingredient.id)
        link = links.get(ingredient.id)
        if link is None:
            link = RecipeIngredient(ingredient=ingredient)
            recipe.ingredients.append(link)
        link.position = position
        link.quantity = line.quantity
        link.unit = line.unit
        link.measure = line.measure
    for ingredient_id, link in links.items():
        if ingredient_id not in wanted_ids:
            recipe.ingredients.remove(link)

    await db.commit()
    reloaded = await crud.recipe.get_recipe(db, recipe.id)
    assert reloaded is not None  # noqa: S101
    return reloaded, created


async def import_recipes(
    db: AsyncSession,
    recipes: AsyncIterable[ExternalRecipe],
    *,
    embedder: Embedder | None = None,
    limit: int | None = None,
    on_progress: Callable[[ImportResult], None] | None = None,
) -> ImportResult:
    """Import recipes one by one and, if an embedder is given, index each right away.

    Each recipe is committed on its own, so an interrupted import keeps what it already
    stored and can simply be re-run; unchanged recipes are not re-embedded.

    Args:
        db: Database session.
        recipes: Recipes to import, e.g. `TheMealDBClient.iter_meals()`.
        embedder: Embedding model; `None` imports without indexing.
        limit: Stop after this many recipes.
        on_progress: Called with the running totals after each recipe.

    Raises:
        LLMServiceError: If indexing fails.
        RecipeSourceError: If the source cannot be read.
    """
    result = ImportResult()
    async for external in recipes:
        if limit is not None and result.imported >= limit:
            break
        recipe, created = await import_recipe(db, external)
        if created:
            result.created += 1
        else:
            result.updated += 1
        if embedder is not None and await index_recipe(db, recipe, embedder):
            result.indexed += 1
        if on_progress is not None:
            on_progress(result)
    return result
