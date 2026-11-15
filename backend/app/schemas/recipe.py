"""Recipe request/response schemas."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.ingredient import IngredientRead


class RecipeIngredientCreate(BaseModel):
    """One ingredient line of a new recipe, with an optional amount."""

    ingredient_id: int
    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = Field(default=None, max_length=32)


class RecipeCreate(BaseModel):
    """Payload for creating a recipe."""

    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1)
    ingredients: list[RecipeIngredientCreate] = Field(default_factory=list)

    @field_validator("title", "instructions")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Strip surrounding whitespace and reject text that is blank after stripping."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("ingredients")
    @classmethod
    def unique_ingredients(
        cls, value: list[RecipeIngredientCreate]
    ) -> list[RecipeIngredientCreate]:
        """Reject recipes that list the same ingredient more than once."""
        ids = [item.ingredient_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("each ingredient may only be listed once per recipe")
        return value


class RecipeIngredientRead(BaseModel):
    """One ingredient line of a recipe as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    ingredient: IngredientRead
    quantity: float | None
    unit: str | None


class RecipeRead(BaseModel):
    """Public representation of a recipe. The raw embedding is intentionally not exposed."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    instructions: str
    ingredients: list[RecipeIngredientRead]
