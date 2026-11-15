"""Source-neutral representation of recipes imported from external datasets."""

from pydantic import BaseModel, Field


class ExternalIngredient(BaseModel):
    """One ingredient line of an imported recipe, with its parsed amount."""

    name: str = Field(min_length=1, max_length=120)
    quantity: float | None = None
    unit: str | None = Field(default=None, max_length=32)


class ExternalRecipe(BaseModel):
    """A recipe from an external source, ready to be upserted into the knowledge base.

    `(source, external_id)` identifies the recipe, which makes imports idempotent.
    """

    source: str = Field(min_length=1, max_length=32)
    external_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1)
    category: str | None = Field(default=None, max_length=64)
    area: str | None = Field(default=None, max_length=64)
    tags: list[str] = Field(default_factory=list)
    thumbnail_url: str | None = Field(default=None, max_length=500)
    source_url: str | None = Field(default=None, max_length=500)
    ingredients: list[ExternalIngredient] = Field(default_factory=list)
