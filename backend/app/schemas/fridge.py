"""Schemas for the fridge ("what's in the fridge") items and recipe suggestions."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.ingredient import IngredientRead, normalize_ingredient_name
from app.schemas.profile import DietProfileData


class FridgeItemCreate(BaseModel):
    """Payload for adding an ingredient to the fridge; unknown ingredients are created."""

    name: str = Field(min_length=1, max_length=120, examples=["chicken thighs"])
    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = Field(default=None, max_length=32)
    expires_on: date | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize like ingredient names and reject blank ones."""
        normalized = normalize_ingredient_name(value)
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized


class FridgeItemUpdate(BaseModel):
    """Partial update of a fridge item; only the fields sent are changed."""

    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = Field(default=None, max_length=32)
    expires_on: date | None = None


class FridgeItemRead(BaseModel):
    """Fridge item as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    ingredient: IngredientRead
    quantity: float | None
    unit: str | None
    expires_on: date | None
    added_at: datetime


class SuggestionRequest(BaseModel):
    """Ask for recipes that can be cooked with what's in the fridge."""

    query: str | None = Field(
        default=None,
        max_length=300,
        description="Optional wish in natural language",
        examples=["something spicy and quick"],
    )
    limit: int = Field(default=3, ge=1, le=8)
    max_missing: int = Field(
        default=4, ge=0, le=15, description="Most ingredients the user would have to buy"
    )
    ingredients: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Use these ingredient names instead of the stored fridge",
    )
    profile: DietProfileData | None = Field(
        default=None, description="Use these constraints instead of the stored diet profile"
    )
    use_llm: bool = Field(default=True, description="Let the LLM rank and adapt the shortlist")

    @field_validator("query")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """Treat a blank query as no query."""
        return (value or "").strip() or None

    @field_validator("ingredients")
    @classmethod
    def normalize_ingredients(cls, value: list[str] | None) -> list[str] | None:
        """Normalize names, dropping blanks and duplicates."""
        if value is None:
            return None
        names = (normalize_ingredient_name(name)[:120] for name in value)
        return list(dict.fromkeys(name for name in names if name))


class Substitution(BaseModel):
    """An ingredient swap applied to a suggested recipe."""

    original: str
    replacement: str
    reason: str = Field(description="Why the swap is made, e.g. 'vegan: milk'")
    source: Literal["llm", "rules"] = Field(
        description="Proposed by the LLM (and validated) or chosen by the substitution rules"
    )


class FridgeSuggestion(BaseModel):
    """A recipe suggestion, adapted to the profile, with what it needs from the fridge."""

    recipe_id: int
    title: str
    category: str | None
    area: str | None
    thumbnail_url: str | None
    score: float = Field(description="Deterministic ranking score in [0, 1]")
    coverage: float = Field(description="Share of the non-staple ingredients already at hand")
    used_ingredients: list[str] = Field(description="Recipe ingredients found in the fridge")
    missing_ingredients: list[str] = Field(description="Ingredients to buy (staples excluded)")
    expiring_used: list[str] = Field(description="Fridge items close to expiry it uses up")
    substitutions: list[Substitution]
    unverified_ingredients: list[str] = Field(
        description="Ingredients the dietary checker doesn't recognise; check them yourself"
    )
    reason: str | None = Field(description="The LLM's explanation; null without the LLM")


class SuggestionResponse(BaseModel):
    """Ranked fridge suggestions and how they were produced."""

    suggestions: list[FridgeSuggestion]
    model: str | None = Field(description="Chat model that ranked the shortlist, if any")
    llm_used: bool
    notes: list[str] = Field(description="Fallbacks taken and LLM output that was rejected")
