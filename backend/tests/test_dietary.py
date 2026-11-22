"""Tests for diets, allergens, the constraint checker and allowed substitutes."""

import pytest
from app.services.dietary import (
    Allergen,
    Diet,
    DietaryProfile,
    check_ingredient,
    find_violations,
    is_verifiable,
)
from app.services.substitutions import allowed_substitutes, candidates_for
from app.services.taxonomy import Tag


def profile(
    *diets: Diet, allergens: tuple[Allergen, ...] = (), excluded: tuple[str, ...] = ()
) -> DietaryProfile:
    """Build a profile from diets and keyword options."""
    return DietaryProfile(diets=frozenset(diets), allergens=frozenset(allergens), excluded=excluded)


@pytest.mark.parametrize(
    ("diet", "allowed", "forbidden"),
    [
        (Diet.VEGETARIAN, ["cheddar cheese", "eggs", "honey"], ["bacon", "salmon", "gelatine"]),
        (Diet.VEGAN, ["tofu", "coconut milk", "vegan butter"], ["butter", "eggs", "honey"]),
        (Diet.PESCATARIAN, ["salmon", "king prawns"], ["chicken", "lard"]),
        (Diet.GLUTEN_FREE, ["rice noodles", "buckwheat", "tamari"], ["flour", "soy sauce"]),
        (Diet.DAIRY_FREE, ["coconut cream", "peanut butter"], ["parmesan", "ghee"]),
        (Diet.KETO, ["cauliflower rice", "butter", "chicken"], ["rice", "sugar", "potatoes"]),
    ],
)
def test_diets(diet: Diet, allowed: list[str], forbidden: list[str]) -> None:
    """Each diet forbids exactly the ingredients it should."""
    p = profile(diet)
    assert [name for name in allowed if check_ingredient(name, p)] == []
    assert [v.ingredient for v in find_violations(forbidden, p)] == forbidden


def test_allergens_and_reasons() -> None:
    """Violations name the diet and the allergy that rule an ingredient out."""
    p = profile(Diet.VEGAN, allergens=(Allergen.PEANUTS, Allergen.MILK))

    violation = check_ingredient("salted butter", p)
    assert violation is not None
    assert violation.tags == {Tag.MILK}
    assert violation.reasons == ("vegan: milk", "allergy: milk")

    peanut = check_ingredient("peanut butter", p)
    assert peanut is not None
    assert peanut.reasons == ("allergy: peanuts",)


def test_excluded_ingredients() -> None:
    """Excluded names match as phrases inside longer names."""
    p = profile(excluded=("mushroom",))
    violation = check_ingredient("wild mushrooms", p)
    assert violation is not None
    assert violation.excluded
    assert violation.reasons == ("excluded by you",)
    assert check_ingredient("mushroom ketchup", p) is not None
    assert check_ingredient("onion", p) is None


def test_empty_profile_allows_everything() -> None:
    """Without constraints nothing is a violation and everything is verifiable."""
    p = profile()
    assert p.is_empty
    assert find_violations(["bacon", "grand marnier"], p) == []
    assert is_verifiable("grand marnier", p)
    assert not is_verifiable("grand marnier", profile(Diet.VEGAN))


def test_from_values_rejects_unknown_diet() -> None:
    """Stored strings are validated against the enums."""
    assert DietaryProfile.from_values(["vegan"], ["soy"], []).diets == {Diet.VEGAN}
    with pytest.raises(ValueError, match="paleo"):
        DietaryProfile.from_values(["paleo"], [], [])


def test_describe() -> None:
    """The prompt line lists every kind of constraint."""
    p = profile(Diet.VEGAN, allergens=(Allergen.TREE_NUTS,), excluded=("coriander",))
    assert p.describe() == "diets: Vegan; allergies: tree nuts; never use: coriander"
    assert profile().describe() == "no dietary restrictions"


def test_candidates_use_longest_phrase() -> None:
    """Peanut butter uses its own entry, not the dairy one for butter."""
    assert candidates_for("unsalted butter") == candidates_for("butter")
    assert candidates_for("peanut butter")[0] == "sunflower seed butter"
    assert candidates_for("grand marnier") == ()


def test_allowed_substitutes_respect_the_whole_profile() -> None:
    """A soy allergy rules out tofu as the vegetarian replacement for chicken."""
    vegetarian = profile(Diet.VEGETARIAN)
    assert allowed_substitutes("chicken breast", vegetarian)[0] == "tofu"
    no_soy = profile(Diet.VEGETARIAN, allergens=(Allergen.SOY,))
    assert "tofu" not in allowed_substitutes("chicken breast", no_soy)


def test_allowed_substitutes_prefer_the_fridge() -> None:
    """Replacements already at hand come first."""
    vegan = profile(Diet.VEGAN)
    assert allowed_substitutes("milk", vegan)[0] == "oat milk"
    assert allowed_substitutes("milk", vegan, prefer=["coconut milk"])[0] == "coconut milk"
