"""Ingredient request/response schemas."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

INGREDIENT_NAME_MAX_LENGTH = 120


def normalize_ingredient_name(name: str) -> str:
    """Collapse whitespace and lowercase, so "Tomato " and "tomato" are the same ingredient.

    Args:
        name: Raw ingredient name.

    Returns:
        The normalized name; empty if `name` is blank.
    """
    return " ".join(name.split()).lower()


class IngredientCreate(BaseModel):
    """Payload for creating an ingredient."""

    name: str = Field(min_length=1, max_length=INGREDIENT_NAME_MAX_LENGTH)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize the name and reject names that are blank after normalization."""
        normalized = normalize_ingredient_name(value)
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized


class IngredientRead(BaseModel):
    """Ingredient as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
