"""Diet profile request/response schemas."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.ingredient import normalize_ingredient_name
from app.services.dietary import Allergen, Diet


class DietProfileData(BaseModel):
    """Dietary constraints: diets, allergens and ingredients the user never wants."""

    model_config = ConfigDict(from_attributes=True)

    diets: list[Diet] = Field(default_factory=list, examples=[["vegetarian"]])
    allergens: list[Allergen] = Field(default_factory=list, examples=[["peanuts"]])
    excluded_ingredients: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Ingredient names to avoid; 'mushroom' also excludes 'wild mushrooms'",
    )

    @field_validator("diets", "allergens")
    @classmethod
    def unique_values(cls, value: list[StrEnum]) -> list[StrEnum]:
        """Drop duplicates, keeping the original order."""
        return list(dict.fromkeys(value))

    @field_validator("excluded_ingredients")
    @classmethod
    def normalize_excluded(cls, value: list[str]) -> list[str]:
        """Normalize names like ingredients are, dropping blanks and duplicates."""
        names = (normalize_ingredient_name(name)[:120] for name in value)
        return list(dict.fromkeys(name for name in names if name))


class LabeledOption(BaseModel):
    """A selectable value with its display label."""

    value: str
    label: str


class DietOption(LabeledOption):
    """A diet and the ingredient properties it rules out."""

    forbids: list[str]


class ProfileOptions(BaseModel):
    """The diets and allergens a profile can contain."""

    diets: list[DietOption]
    allergens: list[LabeledOption]
