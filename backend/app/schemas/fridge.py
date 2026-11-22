"""Schemas for "What's in the Fridge" suggestions (RAG + LLM, constraint-checked)."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from app.schemas.ingredient import normalize_ingredient_name
from app.schemas.search import RecipeSearchFilters


class Allergen(StrEnum):
    """The 14 major allergens of EU Regulation 1169/2011 (milk is called `dairy`)."""

    GLUTEN = "gluten"
    CRUSTACEANS = "crustaceans"
    EGGS = "eggs"
    FISH = "fish"
    PEANUTS = "peanuts"
    SOY = "soy"
    DAIRY = "dairy"
    TREE_NUTS = "tree_nuts"
    CELERY = "celery"
    MUSTARD = "mustard"
    SESAME = "sesame"
    SULPHITES = "sulphites"
    LUPIN = "lupin"
    MOLLUSCS = "molluscs"


class Diet(StrEnum):
    """Dietary patterns that rule out whole food groups."""

    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"


def _clean_names(values: list[str]) -> list[str]:
    """Normalize ingredient names, dropping blanks and duplicates (first one wins)."""
    return list(dict.fromkeys(name for name in map(normalize_ingredient_name, values) if name))


class FridgeRequest(BaseModel):
    """What the user has at hand, plus the restrictions every suggestion must respect."""

    ingredients: list[str] = Field(
        min_length=1, max_length=40, examples=[["chicken", "rice", "garlic", "spinach"]]
    )
    allergens: list[Allergen] = Field(
        default_factory=list, description="Allergens no suggestion may contain"
    )
    diet: Diet | None = None
    avoid: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Other ingredients to keep out (dislikes, intolerances)",
        examples=[["mushroom"]],
    )
    note: str | None = Field(
        default=None, max_length=300, description="Free-text wish, e.g. 'quick weeknight dinner'"
    )
    max_suggestions: int = Field(default=3, ge=1, le=5)
    assume_staples: bool = Field(
        default=True, description="Count salt, pepper, oil, water and sugar as available"
    )
    filters: RecipeSearchFilters = Field(default_factory=RecipeSearchFilters)

    @field_validator("ingredients")
    @classmethod
    def clean_ingredients(cls, value: list[str]) -> list[str]:
        """Normalize names and require at least one non-blank ingredient."""
        cleaned = _clean_names(value)
        if not cleaned:
            raise ValueError("list at least one ingredient")
        return cleaned

    @field_validator("avoid")
    @classmethod
    def clean_avoid(cls, value: list[str]) -> list[str]:
        """Normalize names, dropping blanks and duplicates."""
        return _clean_names(value)

    @field_validator("allergens")
    @classmethod
    def unique_allergens(cls, value: list[Allergen]) -> list[Allergen]:
        """Drop duplicate allergens, keeping the original order."""
        return list(dict.fromkeys(value))

    @field_validator("note")
    @classmethod
    def blank_note_to_none(cls, value: str | None) -> str | None:
        """Strip the note and treat a blank one as missing."""
        return (value or "").strip() or None


class Adaptation(BaseModel):
    """A change to a recipe: swap or drop an ingredient, or a cooking tip."""

    replace: str | None = Field(description="Recipe ingredient replaced; null for a plain tip")
    substitute: str | None = Field(description="What to use instead; null means leave it out")
    note: str


class FridgeSuggestion(BaseModel):
    """A recipe suggested for the fridge; it satisfies every restriction of the request."""

    rank: int = Field(ge=1)
    recipe_id: int
    title: str
    category: str | None
    area: str | None
    thumbnail_url: str | None
    reason: str | None = Field(description="Why it fits (LLM); null when ranked without the LLM")
    used_ingredients: list[str] = Field(description="Recipe ingredients you already have")
    missing_ingredients: list[str] = Field(description="Ingredients you'd still need")
    adaptations: list[Adaptation]
    coverage: float = Field(
        ge=0, le=1, description="Share of the (adapted) recipe's non-staple ingredients you have"
    )
    similarity: float = Field(description="Retrieval similarity of the recipe to the fridge")


class ExcludedRecipe(BaseModel):
    """A retrieved recipe dropped because it breaks the request's restrictions."""

    recipe_id: int
    title: str
    reasons: list[str] = Field(examples=[["butter: contains dairy"]])


class FridgeResponse(BaseModel):
    """Ranked, constraint-checked suggestions for what's in the fridge."""

    ingredients: list[str] = Field(description="The normalized fridge contents")
    suggestions: list[FridgeSuggestion]
    excluded: list[ExcludedRecipe]
    model: str | None = Field(description="Chat model that ranked the suggestions; null if none")
