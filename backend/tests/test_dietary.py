"""Unit tests for ingredient matching and the allergen / diet rules."""

import pytest
from app.schemas import Allergen, Diet
from app.services.dietary import DietaryConstraints, Violation
from app.services.ingredient_text import find_ingredient, mentions, same_ingredient, tokenize


def test_tokenize_normalizes_case_accents_and_plurals() -> None:
    """Tokens are lowercase, accent-free and singular; single letters are dropped."""
    assert tokenize("Crème Fraîche") == ("creme", "fraiche")
    assert tokenize("Tomatoes, Berries & Eggs") == ("tomato", "berry", "egg")
    assert tokenize("Goat's Cheese") == ("goat", "cheese")


def test_mentions_matches_whole_words_only() -> None:
    """A phrase must appear as whole, contiguous words."""
    assert mentions("Egg Yolks", "egg")
    assert not mentions("Eggplant", "egg")
    assert mentions("chicken breast", "Chicken")
    assert not mentions("chicken", "chicken breast")


def test_same_ingredient_and_find_ingredient() -> None:
    """Either name containing the other counts as the same ingredient."""
    assert same_ingredient("chicken", "chicken thighs")
    assert same_ingredient("chicken thighs", "chicken")
    assert not same_ingredient("rice", "spinach")
    assert find_ingredient(["Spinach", "Basmati Rice"], "rice") == "Basmati Rice"
    assert find_ingredient(["Spinach"], "rice") is None


@pytest.mark.parametrize(
    ("ingredient", "allergen"),
    [
        ("Unsalted Butter", Allergen.DAIRY),
        ("Parmesan", Allergen.DAIRY),
        ("Crème Fraîche", Allergen.DAIRY),
        ("Plain Flour", Allergen.GLUTEN),
        ("Soy Sauce", Allergen.GLUTEN),
        ("Soy Sauce", Allergen.SOY),
        ("Egg Yolks", Allergen.EGGS),
        ("King Prawns", Allergen.CRUSTACEANS),
        ("Worcestershire Sauce", Allergen.FISH),
        ("Pine Nuts", Allergen.TREE_NUTS),
        ("Peanut Butter", Allergen.PEANUTS),
        ("Tahini", Allergen.SESAME),
        ("Red Wine", Allergen.SULPHITES),
    ],
)
def test_allergen_triggers(ingredient: str, allergen: Allergen) -> None:
    """Common ingredients are recognized as containing their allergen."""
    assert DietaryConstraints(allergens=frozenset({allergen})).reasons(ingredient)


@pytest.mark.parametrize(
    ("ingredient", "allergen"),
    [
        ("Peanut Butter", Allergen.DAIRY),
        ("Coconut Milk", Allergen.DAIRY),
        ("Butter Beans", Allergen.DAIRY),
        ("Vegan Butter", Allergen.DAIRY),
        ("Eggplant", Allergen.EGGS),
        ("Rice Flour", Allergen.GLUTEN),
        ("Gluten-free Flour", Allergen.GLUTEN),
        ("Buckwheat", Allergen.GLUTEN),
        ("Nutmeg", Allergen.TREE_NUTS),
        ("Butternut Squash", Allergen.TREE_NUTS),
        ("Oyster Mushrooms", Allergen.MOLLUSCS),
    ],
)
def test_allergen_lookalikes_are_exempt(ingredient: str, allergen: Allergen) -> None:
    """Lookalike names don't trigger the allergen."""
    assert not DietaryConstraints(allergens=frozenset({allergen})).reasons(ingredient)


@pytest.mark.parametrize(
    ("diet", "allowed", "forbidden"),
    [
        (Diet.VEGETARIAN, ["Kidney Beans", "Eggs", "Cheddar"], ["Chicken Stock", "Bacon", "Cod"]),
        (Diet.PESCATARIAN, ["Salmon", "Prawns", "Eggs"], ["Beef Mince", "Gelatine"]),
        (Diet.VEGAN, ["Tofu", "Coconut Milk", "Beef Tomato"], ["Honey", "Milk", "Eggs", "Tuna"]),
    ],
)
def test_diet_rules(diet: Diet, allowed: list[str], forbidden: list[str]) -> None:
    """Each diet forbids exactly its food groups."""
    constraints = DietaryConstraints(diet=diet)
    for ingredient in allowed:
        assert not constraints.reasons(ingredient), ingredient
    for ingredient in forbidden:
        assert constraints.reasons(ingredient), ingredient


def test_diet_exemption_doesnt_leak_across_groups() -> None:
    """A "dairy free" label doesn't make meat vegetarian."""
    assert DietaryConstraints(diet=Diet.VEGETARIAN).reasons("dairy free chicken")


def test_violations_cover_avoided_ingredients_and_describe() -> None:
    """Avoided ingredients match by word; every broken rule is reported once."""
    constraints = DietaryConstraints(
        allergens=frozenset({Allergen.DAIRY}), diet=Diet.VEGAN, avoid=("mushroom",)
    )

    violations = constraints.violations(["Butter", "Chestnut Mushrooms", "Rice"])

    assert violations == [
        Violation("Butter", "contains dairy"),
        Violation("Butter", "not vegan (dairy)"),
        Violation("Chestnut Mushrooms", "you avoid mushroom"),
    ]
    assert str(violations[0]) == "Butter: contains dairy"
    assert constraints.describe() == "vegan; allergic to: dairy; avoids: mushroom"


def test_empty_constraints_allow_everything() -> None:
    """Without restrictions nothing is a violation."""
    constraints = DietaryConstraints()
    assert constraints.is_empty
    assert constraints.violations(["Bacon", "Butter"]) == []
    assert constraints.describe() == "none"
