"""Ingredient request/response schemas."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IngredientCreate(BaseModel):
    """Payload for creating an ingredient."""

    name: str = Field(min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Trim and lowercase so "Tomato " and "tomato" can't coexist."""
        normalized = " ".join(value.split()).lower()
        if not normalized:
            raise ValueError("name must not be blank")
        return normalized


class IngredientRead(BaseModel):
    """Ingredient as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
