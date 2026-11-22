"""Tests for the deterministic fridge matching, adaptation and scoring."""

from datetime import date

from app.models import Ingredient, Recipe, RecipeIngredient
from app.services.dietary import Allergen, Diet, DietaryProfile
from app.services.fridge_ranking import (
    Adaptation,
    Fridge,
    FridgeEntry,
    ProposedSwap,
    adapt,
    evaluate,
    score,
    shortlist,
)

TODAY = date(2026, 11, 22)
VEGAN = DietaryProfile(diets=frozenset({Diet.VEGAN}))
NONE = DietaryProfile()


def make_recipe(recipe_id: int, title: str, *ingredients: str) -> Recipe:
    """Build a transient recipe with the given ingredient names."""
    return Recipe(
        id=recipe_id,
        title=title,
        instructions="Cook.",
        ingredients=[RecipeIngredient(ingredient=Ingredient(name=name)) for name in ingredients],
    )


def make_fridge(*names: str, expiring: tuple[str, ...] = ()) -> Fridge:
    """Build a fridge; names in `expiring` expire tomorrow."""
    entries = [
        FridgeEntry(name, date(2026, 11, 23) if name in expiring else None) for name in names
    ]
    return Fridge.build(entries, today=TODAY, expiring_days=3)


def test_fridge_build_sorts_out_expired_and_expiring() -> None:
    """Expired items are dropped, items within the horizon are flagged."""
    fridge = Fridge.build(
        [
            FridgeEntry("milk", date(2026, 11, 20)),
            FridgeEntry("spinach", date(2026, 11, 25)),
            FridgeEntry("rice", date(2026, 12, 30)),
            FridgeEntry("salt"),
        ],
        today=TODAY,
        expiring_days=3,
    )
    assert fridge.names == ["spinach", "rice", "salt"]
    assert fridge.expired == ("milk",)
    assert fridge.expiring == {"spinach"}


def test_adapt_swaps_violations_with_rules() -> None:
    """Each forbidden ingredient gets the best allowed replacement, in recipe order."""
    adaptation = adapt(["chicken", "rice", "butter"], VEGAN, make_fridge("rice"))
    assert adaptation is not None
    assert [(s.original, s.replacement, s.source) for s in adaptation.substitutions] == [
        ("chicken", "tofu", "rules"),
        ("butter", "olive oil", "rules"),
    ]
    assert adaptation.substitutions[1].reason == "vegan: milk"


def test_adapt_prefers_replacements_in_the_fridge() -> None:
    """Chickpeas in the fridge beat the default tofu."""
    adaptation = adapt(["chicken"], VEGAN, make_fridge("chickpeas"))
    assert adaptation is not None
    assert adaptation.substitutions[0].replacement == "chickpeas"


def test_adapt_fails_without_allowed_substitute() -> None:
    """A violation the table can't fix makes the recipe unusable."""
    assert adapt(["black pudding"], VEGAN, make_fridge()) is None


def test_adapt_validates_llm_swaps() -> None:
    """LLM swaps are applied only if they are safe and useful; the rules fill the gaps."""
    proposed = [
        ProposedSwap("butter", "ghee"),  # still dairy
        ProposedSwap("chicken", "Smoked Tofu"),  # fine, normalized
        ProposedSwap("lobster", "tofu"),  # not in the recipe
        ProposedSwap("rice", "quinoa"),  # neither a fix nor from the fridge
        ProposedSwap("honey", "grand marnier"),  # unknown to the checker
    ]
    adaptation = adapt(["chicken", "butter", "rice", "honey"], VEGAN, make_fridge(), proposed)

    assert adaptation is not None
    swaps = {s.original: (s.replacement, s.source) for s in adaptation.substitutions}
    assert swaps == {
        "chicken": ("smoked tofu", "llm"),
        "butter": ("olive oil", "rules"),
        "honey": ("maple syrup", "rules"),
    }
    assert len(adaptation.rejected) == 4
    assert adaptation.rejected[0] == "butter -> ghee: vegan: milk"


def test_adapt_accepts_fridge_swaps_for_missing_ingredients() -> None:
    """Swapping a missing ingredient for one at hand is allowed without constraints."""
    adaptation = adapt(
        ["kale", "rice"], NONE, make_fridge("spinach", "rice"), [ProposedSwap("kale", "spinach")]
    )
    assert adaptation is not None
    assert adaptation.substitutions[0].reason == "uses what's in your fridge"


def test_evaluate_counts_coverage_usage_and_expiry() -> None:
    """Staples are ignored; substitutions are evaluated, not the originals."""
    fridge = make_fridge("tofu", "rice", "spinach", "lemon", expiring=("spinach",))
    adaptation = adapt(["chicken thighs", "rice", "spinach", "ginger", "salt"], VEGAN, fridge)
    assert adaptation is not None

    evaluation = evaluate(
        ["chicken thighs", "rice", "spinach", "ginger", "salt"], adaptation, fridge, VEGAN
    )

    assert evaluation.used == ["tofu", "rice", "spinach"]
    assert evaluation.missing == ["ginger"]
    assert evaluation.coverage == 0.75
    assert evaluation.usage == 0.75
    assert evaluation.expiring_used == ["spinach"]
    assert evaluation.expiring_share == 1.0
    assert evaluation.unverified == []


def test_evaluate_counts_a_shared_replacement_once() -> None:
    """Two meats both swapped to lentils make one missing ingredient."""
    adaptation = adapt(["ground beef", "ground pork", "onion"], VEGAN, make_fridge("onion"))
    assert adaptation is not None
    evaluation = evaluate(
        ["ground beef", "ground pork", "onion"], adaptation, make_fridge("onion"), VEGAN
    )
    assert evaluation.missing == ["lentils"]
    assert evaluation.coverage == 0.5


def test_evaluate_flags_unverified_ingredients() -> None:
    """Names the taxonomy doesn't know are reported under a constrained profile."""
    evaluation = evaluate(["grand marnier"], Adaptation([]), make_fridge(), VEGAN)
    assert evaluation.unverified == ["grand marnier"]


def test_score_is_bounded() -> None:
    """A perfect match scores 1, the semantic part is clamped."""
    fridge = make_fridge("rice", expiring=("rice",))
    evaluation = evaluate(["rice"], Adaptation([]), fridge, NONE)
    assert score(evaluation, semantic=1.7) == 1.0
    assert score(evaluation, semantic=-0.3) == 0.8


def test_shortlist_filters_and_orders() -> None:
    """Unadaptable recipes, recipes without fridge items and too many missing are dropped."""
    fridge = make_fridge("rice", "spinach", "garlic", expiring=("spinach",))
    recipes = [
        make_recipe(1, "Garlic Rice", "rice", "garlic", "salt"),
        make_recipe(2, "Spinach Rice", "rice", "spinach", "garlic", "onion"),
        make_recipe(3, "Black Pudding", "black pudding", "rice"),
        make_recipe(4, "Cake", "flour", "sugar", "eggs"),
        make_recipe(5, "Big Curry", "rice", "a", "b", "c", "d", "e"),
    ]

    result = shortlist(recipes, fridge, VEGAN, semantic={2: 0.5}, max_missing=2)

    assert [c.recipe.title for c in result] == ["Spinach Rice", "Garlic Rice"]
    assert result[0].evaluation.missing == ["onion"]


def test_shortlist_respects_allergies() -> None:
    """An allergen is swapped out, or the recipe is dropped if it can't be."""
    no_nuts = DietaryProfile(allergens=frozenset({Allergen.PEANUTS}))
    fridge = make_fridge("noodles", "peanut butter")
    recipes = [make_recipe(1, "Satay Noodles", "noodles", "peanut butter", "lime")]

    (candidate,) = shortlist(recipes, fridge, no_nuts, semantic={}, max_missing=3)

    assert [s.replacement for s in candidate.adaptation.substitutions] == ["sunflower seed butter"]
    assert "peanut butter" not in candidate.evaluation.used
