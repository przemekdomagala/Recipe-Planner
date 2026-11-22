"""Diets, allergens and the constraint checker built on the ingredient taxonomy.

A `DietaryProfile` turns the user's diets, allergens and excluded ingredients into a set of
forbidden taxonomy tags; `find_violations` reports every ingredient that breaks it. The
checker is deterministic and is the final authority over anything the LLM proposes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from app.services.taxonomy import Tag, classify, contains_phrase, tokenize


class Diet(StrEnum):
    """A diet the user follows."""

    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    KETO = "keto"


class Allergen(StrEnum):
    """One of the 14 allergens that EU food law requires to be declared."""

    GLUTEN = Tag.GLUTEN.value
    CRUSTACEANS = Tag.CRUSTACEANS.value
    EGGS = Tag.EGGS.value
    FISH = Tag.FISH.value
    PEANUTS = Tag.PEANUTS.value
    SOY = Tag.SOY.value
    MILK = Tag.MILK.value
    TREE_NUTS = Tag.TREE_NUTS.value
    CELERY = Tag.CELERY.value
    MUSTARD = Tag.MUSTARD.value
    SESAME = Tag.SESAME.value
    SULPHITES = Tag.SULPHITES.value
    LUPIN = Tag.LUPIN.value
    MOLLUSCS = Tag.MOLLUSCS.value


_NOT_VEGETARIAN = frozenset(
    {Tag.MEAT, Tag.FISH, Tag.CRUSTACEANS, Tag.MOLLUSCS, Tag.ANIMAL_BYPRODUCT}
)

DIET_FORBIDS: dict[Diet, frozenset[Tag]] = {
    Diet.VEGETARIAN: _NOT_VEGETARIAN,
    Diet.VEGAN: _NOT_VEGETARIAN | {Tag.MILK, Tag.EGGS, Tag.HONEY},
    Diet.PESCATARIAN: frozenset({Tag.MEAT, Tag.ANIMAL_BYPRODUCT}),
    Diet.GLUTEN_FREE: frozenset({Tag.GLUTEN}),
    Diet.DAIRY_FREE: frozenset({Tag.MILK}),
    Diet.KETO: frozenset({Tag.HIGH_CARB}),
}

DIET_LABELS: dict[Diet, str] = {
    Diet.VEGETARIAN: "Vegetarian",
    Diet.VEGAN: "Vegan",
    Diet.PESCATARIAN: "Pescatarian",
    Diet.GLUTEN_FREE: "Gluten-free",
    Diet.DAIRY_FREE: "Dairy-free",
    Diet.KETO: "Keto",
}

ALLERGEN_LABELS: dict[Allergen, str] = {
    Allergen.GLUTEN: "Gluten",
    Allergen.CRUSTACEANS: "Crustaceans",
    Allergen.EGGS: "Eggs",
    Allergen.FISH: "Fish",
    Allergen.PEANUTS: "Peanuts",
    Allergen.SOY: "Soy",
    Allergen.MILK: "Milk",
    Allergen.TREE_NUTS: "Tree nuts",
    Allergen.CELERY: "Celery",
    Allergen.MUSTARD: "Mustard",
    Allergen.SESAME: "Sesame",
    Allergen.SULPHITES: "Sulphites",
    Allergen.LUPIN: "Lupin",
    Allergen.MOLLUSCS: "Molluscs",
}


@dataclass(frozen=True, slots=True)
class DietaryProfile:
    """The constraints a recipe must satisfy."""

    diets: frozenset[Diet] = frozenset()
    allergens: frozenset[Allergen] = frozenset()
    excluded: tuple[str, ...] = ()
    _excluded_tokens: tuple[tuple[str, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Pre-tokenize the excluded ingredient names."""
        tokens = tuple(t for t in (tokenize(name) for name in self.excluded) if t)
        object.__setattr__(self, "_excluded_tokens", tokens)

    @classmethod
    def from_values(
        cls, diets: Iterable[str], allergens: Iterable[str], excluded: Iterable[str]
    ) -> DietaryProfile:
        """Build a profile from stored or requested string values.

        Raises:
            ValueError: If a diet or allergen value is unknown.
        """
        return cls(
            diets=frozenset(Diet(value) for value in diets),
            allergens=frozenset(Allergen(value) for value in allergens),
            excluded=tuple(excluded),
        )

    @property
    def is_empty(self) -> bool:
        """Whether the profile has no constraints at all."""
        return not (self.diets or self.allergens or self.excluded)

    @property
    def forbidden_tags(self) -> frozenset[Tag]:
        """All taxonomy tags excluded by the diets and allergens."""
        tags = {Tag(allergen.value) for allergen in self.allergens}
        for diet in self.diets:
            tags |= DIET_FORBIDS[diet]
        return frozenset(tags)

    def is_excluded(self, name: str) -> bool:
        """Whether the user excluded this ingredient ("mushroom" excludes "wild mushrooms")."""
        tokens = tokenize(name)
        return any(contains_phrase(tokens, phrase) for phrase in self._excluded_tokens)

    def reasons(self, tags: Iterable[Tag]) -> list[str]:
        """Explain which diets and allergens forbid `tags`, e.g. `["vegan: milk"]`."""
        tags = set(tags)
        reasons = [
            f"{DIET_LABELS[diet].lower()}: {', '.join(sorted(DIET_FORBIDS[diet] & tags))}"
            for diet in sorted(self.diets)
            if DIET_FORBIDS[diet] & tags
        ]
        reasons += [
            f"allergy: {ALLERGEN_LABELS[allergen].lower()}"
            for allergen in sorted(self.allergens)
            if Tag(allergen.value) in tags
        ]
        return reasons

    def describe(self) -> str:
        """Render the constraints as one line for prompts and notes."""
        parts = []
        if self.diets:
            parts.append("diets: " + ", ".join(DIET_LABELS[d] for d in sorted(self.diets)))
        if self.allergens:
            labels = ", ".join(ALLERGEN_LABELS[a].lower() for a in sorted(self.allergens))
            parts.append("allergies: " + labels)
        if self.excluded:
            parts.append("never use: " + ", ".join(self.excluded))
        return "; ".join(parts) or "no dietary restrictions"


@dataclass(frozen=True, slots=True)
class Violation:
    """An ingredient that breaks the profile, and why."""

    ingredient: str
    tags: frozenset[Tag]
    excluded: bool
    reasons: tuple[str, ...]


def check_ingredient(name: str, profile: DietaryProfile) -> Violation | None:
    """Check one ingredient against the profile.

    Args:
        name: Ingredient name.
        profile: The constraints to enforce.

    Returns:
        The violation, or `None` if the ingredient is allowed.
    """
    tags = classify(name).tags & profile.forbidden_tags
    excluded = profile.is_excluded(name)
    if not tags and not excluded:
        return None
    reasons = profile.reasons(tags) + (["excluded by you"] if excluded else [])
    return Violation(ingredient=name, tags=tags, excluded=excluded, reasons=tuple(reasons))


def find_violations(names: Iterable[str], profile: DietaryProfile) -> list[Violation]:
    """Return the violation of every ingredient in `names` that breaks the profile."""
    if profile.is_empty:
        return []
    return [v for v in (check_ingredient(name, profile) for name in names) if v is not None]


def is_verifiable(name: str, profile: DietaryProfile) -> bool:
    """Whether the checker can vouch for the ingredient under this profile.

    An ingredient no taxonomy rule recognises can't be verified, unless the profile has no
    constraints that could be violated.
    """
    return profile.is_empty or classify(name).known
