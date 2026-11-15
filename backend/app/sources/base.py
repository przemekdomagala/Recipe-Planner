"""Source-independent representation of a recipe fetched from an external provider."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExternalIngredient:
    """One ingredient line as published by the source."""

    name: str
    measure: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalRecipe:
    """A recipe from an external source, before it is mapped onto the ORM models.

    `(source, external_id)` uniquely identifies the recipe across imports.
    """

    source: str
    external_id: str
    title: str
    instructions: str
    category: str | None = None
    area: str | None = None
    tags: tuple[str, ...] = ()
    image_url: str | None = None
    source_url: str | None = None
    ingredients: tuple[ExternalIngredient, ...] = field(default_factory=tuple)
