"""Deterministic allergen and diet checks on ingredient names.

This is the safety net of the fridge suggestions: the LLM proposes, these rules decide.
Each food group is a set of trigger phrases plus exemptions that clear an ingredient for
that group ("peanut butter" is not dairy, "kidney beans" are not meat). Matching is done
on normalized tokens (`app.services.ingredient_text`), so "Egg Yolks" contains eggs and
"Eggplant" doesn't.

The lexicon errs on the side of caution (e.g. soy sauce counts as gluten), but it can't
know every product: an ingredient it doesn't recognize passes. It's a filter for
suggestions, not a substitute for reading labels.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

from app.schemas.fridge import Allergen, Diet, FridgeRequest
from app.services.ingredient_text import Tokens, contains_phrase, tokenize


@dataclass(frozen=True, slots=True)
class _FoodGroup:
    """Trigger phrases for a food group and phrases that exempt an ingredient from it."""

    label: str
    triggers: tuple[Tokens, ...]
    exemptions: tuple[Tokens, ...] = ()

    @classmethod
    def of(cls, label: str, triggers: str, exemptions: str = "") -> _FoodGroup:
        """Build a group from comma-separated phrase lists."""

        def phrases(text: str) -> tuple[Tokens, ...]:
            """Tokenize each non-blank comma-separated phrase."""
            return tuple(tokens for part in text.split(",") if (tokens := tokenize(part)))

        return cls(label, phrases(triggers), phrases(exemptions))

    def matches(self, tokens: Tokens) -> bool:
        """Whether an ingredient (as tokens) belongs to this group."""
        if any(contains_phrase(tokens, phrase) for phrase in self.exemptions):
            return False
        return any(contains_phrase(tokens, phrase) for phrase in self.triggers)


# Labels that mark a product as free of animal ingredients.
_VEGAN_LABELS = "vegan, plant based"

_ALLERGENS: Final[dict[Allergen, _FoodGroup]] = {
    Allergen.GLUTEN: _FoodGroup.of(
        "gluten",
        "wheat, oat, flour, bread, breadcrumb, panko, pasta, spaghetti, penne, macaroni, fusilli,"
        " linguine, tagliatelle, fettuccine, lasagne, lasagna, rigatoni, farfalle, orzo,"
        " noodle, udon, ramen, couscous, bulgur, semolina, barley, rye, spelt, seitan,"
        " tortilla, pita, naan, pastry, filo, phyllo, biscuit, cracker, cake, croissant,"
        " baguette, brioche, bun, dumpling, wonton, gnocchi, ravioli, tortellini, beer,"
        " malt, suet, soy sauce, teriyaki, hoisin",
        "gluten free, rice flour, corn flour, almond flour, coconut flour, chickpea flour,"
        " gram flour, potato flour, tapioca flour, cassava flour, rice noodle,"
        " rice vermicelli, glass noodle, corn tortilla, rice cake, rice cracker",
    ),
    Allergen.CRUSTACEANS: _FoodGroup.of(
        "crustaceans",
        "prawn, shrimp, crab, lobster, crayfish, crawfish, langoustine, scampi",
    ),
    Allergen.MOLLUSCS: _FoodGroup.of(
        "molluscs",
        "mussel, clam, oyster, scallop, squid, calamari, octopus, cuttlefish, snail,"
        " escargot, cockle, whelk",
        "oyster mushroom",
    ),
    Allergen.FISH: _FoodGroup.of(
        "fish",
        "fish, salmon, tuna, cod, haddock, hake, anchovy, sardine, pilchard, mackerel, trout,"
        " tilapia, halibut, sea bass, monkfish, pollock, herring, kipper, snapper, swordfish,"
        " plaice, eel, caviar, roe, bonito, dashi, worcestershire",
    ),
    Allergen.EGGS: _FoodGroup.of(
        "eggs",
        "egg, mayonnaise, mayo, meringue, aioli, hollandaise",
        f"egg free, {_VEGAN_LABELS}",
    ),
    Allergen.PEANUTS: _FoodGroup.of("peanuts", "peanut, groundnut, satay"),
    Allergen.TREE_NUTS: _FoodGroup.of(
        "tree nuts",
        "nut, almond, cashew, walnut, pecan, hazelnut, pistachio, macadamia, brazil nut,"
        " pine nut, marzipan, praline, nutella, frangipane, pesto",
        "nut free",
    ),
    Allergen.SOY: _FoodGroup.of(
        "soy",
        "soy, soya, soybean, tofu, tempeh, edamame, miso, tamari, teriyaki, hoisin, bean curd",
    ),
    Allergen.DAIRY: _FoodGroup.of(
        "dairy",
        "milk, butter, buttermilk, cream, cheese, yogurt, yoghurt, ghee, creme fraiche, curd,"
        " parmesan, mozzarella, cheddar, feta, ricotta, mascarpone, brie, camembert, gouda,"
        " gruyere, halloumi, paneer, pecorino, stilton, custard, whey, casein, quark, kefir,"
        " fromage frais",
        "coconut milk, coconut cream, coconut yogurt, almond milk, oat milk, soy milk,"
        " soya milk, rice milk, peanut butter, cocoa butter, almond butter, cashew butter,"
        f" nut butter, butter bean, cream of tartar, bean curd, dairy free, {_VEGAN_LABELS}",
    ),
    Allergen.CELERY: _FoodGroup.of("celery", "celery, celeriac"),
    Allergen.MUSTARD: _FoodGroup.of("mustard", "mustard, dijon"),
    Allergen.SESAME: _FoodGroup.of("sesame", "sesame, tahini, halva, hummus"),
    Allergen.SULPHITES: _FoodGroup.of(
        "sulphites",
        "wine, sherry, vermouth, marsala, balsamic, prosecco, champagne, cider, dried apricot",
    ),
    Allergen.LUPIN: _FoodGroup.of("lupin", "lupin, lupine, lupini"),
}

_MEAT: Final = _FoodGroup.of(
    "meat",
    "beef, pork, chicken, lamb, mutton, veal, goat, venison, rabbit, duck, turkey, goose,"
    " pheasant, quail, partridge, bacon, ham, gammon, sausage, chorizo, salami, pepperoni,"
    " prosciutto, pancetta, mince, steak, brisket, sirloin, oxtail, kidney, liver, offal,"
    " meatball, burger, pate, foie gras, black pudding, haggis, gelatin, gelatine, lard, suet,"
    " dripping, bone broth",
    "kidney bean, beef tomato, goat cheese, goat milk, duck egg, vegetarian, veggie, meat free,"
    f" {_VEGAN_LABELS}",
)
_HONEY: Final = _FoodGroup.of("honey", "honey, beeswax, royal jelly")

_SEAFOOD = (
    _ALLERGENS[Allergen.FISH],
    _ALLERGENS[Allergen.CRUSTACEANS],
    _ALLERGENS[Allergen.MOLLUSCS],
)
_DIET_EXCLUDES: Final[dict[Diet, tuple[_FoodGroup, ...]]] = {
    Diet.PESCATARIAN: (_MEAT,),
    Diet.VEGETARIAN: (_MEAT, *_SEAFOOD),
    Diet.VEGAN: (
        _MEAT,
        *_SEAFOOD,
        _ALLERGENS[Allergen.DAIRY],
        _ALLERGENS[Allergen.EGGS],
        _HONEY,
    ),
}


@dataclass(frozen=True, slots=True)
class Violation:
    """An ingredient that breaks a restriction, and why."""

    ingredient: str
    reason: str

    def __str__(self) -> str:
        """Human-readable form, e.g. `butter: contains dairy`."""
        return f"{self.ingredient}: {self.reason}"


@dataclass(frozen=True, slots=True)
class DietaryConstraints:
    """Allergens, a diet and avoided ingredients that every suggestion must respect."""

    allergens: frozenset[Allergen] = frozenset()
    diet: Diet | None = None
    avoid: tuple[str, ...] = ()
    _avoid_tokens: tuple[tuple[str, Tokens], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Pre-tokenize the avoided ingredients."""
        tokens = tuple((name, tokenize(name)) for name in self.avoid)
        object.__setattr__(self, "_avoid_tokens", tuple(item for item in tokens if item[1]))

    @classmethod
    def from_request(cls, request: FridgeRequest) -> DietaryConstraints:
        """The restrictions stated in a fridge request."""
        return cls(frozenset(request.allergens), request.diet, tuple(request.avoid))

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to enforce."""
        return not (self.allergens or self.diet or self._avoid_tokens)

    def describe(self) -> str:
        """One-line summary for the LLM prompt, e.g. `vegan; allergic to: gluten`."""
        parts = []
        if self.diet:
            parts.append(self.diet.value)
        if self.allergens:
            labels = sorted(_ALLERGENS[allergen].label for allergen in self.allergens)
            parts.append("allergic to: " + ", ".join(labels))
        if self.avoid:
            parts.append("avoids: " + ", ".join(self.avoid))
        return "; ".join(parts) or "none"

    def reasons(self, ingredient: str) -> list[str]:
        """Why `ingredient` is not allowed; empty if it is."""
        tokens = tokenize(ingredient)
        found = [
            f"contains {_ALLERGENS[allergen].label}"
            for allergen in sorted(self.allergens)
            if _ALLERGENS[allergen].matches(tokens)
        ]
        if self.diet:
            found.extend(
                f"not {self.diet.value} ({group.label})"
                for group in _DIET_EXCLUDES[self.diet]
                if group.matches(tokens)
            )
        found.extend(
            f"you avoid {name}"
            for name, phrase in self._avoid_tokens
            if contains_phrase(tokens, phrase)
        )
        return list(dict.fromkeys(found))

    def violations(self, ingredients: Iterable[str]) -> list[Violation]:
        """Every (ingredient, reason) pair that breaks the restrictions, in input order."""
        if self.is_empty:
            return []
        return [
            Violation(ingredient, reason)
            for ingredient in ingredients
            for reason in self.reasons(ingredient)
        ]
