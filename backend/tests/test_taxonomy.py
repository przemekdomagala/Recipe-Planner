"""Tests for the ingredient taxonomy and the substitution table."""

import pytest
from app.services.substitutions import SUBSTITUTES
from app.services.taxonomy import Tag, classify, is_staple, same_ingredient, tokenize

T = Tag


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Free-range Eggs, beaten", ("free", "range", "egg", "beaten")),
        ("Crème Fraîche", ("creme", "fraiche")),
        ("Cherry Tomatoes", ("cherry", "tomato")),
        ("Bay Leaves", ("bay", "leaf")),
        ("Anchovies", ("anchovy",)),
        ("Peaches", ("peach",)),
        ("Hummus", ("hummus",)),
    ],
)
def test_tokenize(name: str, expected: tuple[str, ...]) -> None:
    """Names are lowercased, stripped of accents and punctuation and singularized."""
    assert tokenize(name) == expected


@pytest.mark.parametrize(
    ("name", "tags"),
    [
        ("butter", {T.MILK}),
        ("unsalted butter", {T.MILK}),
        ("chicken thighs", {T.MEAT}),
        ("chicken stock", {T.MEAT}),
        ("thai fish sauce", {T.FISH}),
        ("worcestershire sauce", {T.FISH}),
        ("king prawns", {T.CRUSTACEANS}),
        ("oyster sauce", {T.MOLLUSCS, T.GLUTEN}),
        ("soy sauce", {T.SOY, T.GLUTEN}),
        ("spaghetti", {T.GLUTEN, T.HIGH_CARB}),
        ("egg noodles", {T.EGGS, T.GLUTEN, T.HIGH_CARB}),
        ("clear honey", {T.HONEY, T.HIGH_CARB}),
        ("gelatine leafs", {T.ANIMAL_BYPRODUCT}),
        ("thai red curry paste", {T.CRUSTACEANS}),
        ("sesame seed burger buns", {T.SESAME, T.GLUTEN, T.HIGH_CARB}),
        ("dijon mustard", {T.MUSTARD}),
        ("celery salt", {T.CELERY}),
        ("dry white wine", {T.SULPHITES}),
        ("basmati rice", {T.HIGH_CARB}),
    ],
)
def test_classify_tags(name: str, tags: set[Tag]) -> None:
    """Keyword rules attach the expected allergen and diet tags."""
    result = classify(name)
    assert result.known
    assert result.tags == tags


@pytest.mark.parametrize(
    ("name", "tags"),
    [
        # A longer phrase claims its words, so the shorter rule doesn't fire.
        ("peanut butter", {T.PEANUTS}),
        ("sunflower seed butter", set()),
        ("butter beans", {T.HIGH_CARB}),
        ("coconut milk", set()),
        ("cream of tartar", set()),
        ("egg plants", set()),
        ("oyster mushrooms", set()),
        ("beef tomatoes", set()),
        ("ground nutmeg", set()),
        ("ground nut oil", {T.PEANUTS}),
        ("buckwheat", {T.HIGH_CARB}),
        ("glutinous rice", {T.HIGH_CARB}),
        ("corn flour", {T.HIGH_CARB}),
        ("stir-fry vegetables", set()),
        # Qualifiers remove tags.
        ("gluten free pasta", {T.HIGH_CARB}),
        ("vegan butter", set()),
        ("vegan curry paste", set()),
    ],
)
def test_classify_tricky_names(name: str, tags: set[Tag]) -> None:
    """Overrides and qualifiers avoid the classic false positives."""
    assert classify(name).tags == tags


def test_unknown_names_are_not_assumed_safe() -> None:
    """A name no rule matches is reported as unknown, not as tag-free and known."""
    result = classify("grand marnier")
    assert not result.known
    assert result.tags == frozenset()


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("chicken", "chicken thighs", True),
        ("onion", "red onions", True),
        ("eggs", "free-range egg, beaten", True),
        ("chopped tomatoes", "tomatoes", True),
        ("garlic", "garlic clove", True),
        ("coriander", "cilantro", True),
        ("chicken", "chicken stock", False),
        ("butter", "peanut butter", False),
        ("rice", "rice vinegar", False),
        ("milk", "coconut milk", False),
    ],
)
def test_same_ingredient(a: str, b: str, same: bool) -> None:
    """Descriptors are ignored, but different products never match."""
    assert same_ingredient(a, b) is same


@pytest.mark.parametrize(
    ("name", "staple"),
    [
        ("salt", True),
        ("ground black pepper", True),
        ("extra virgin olive oil", True),
        ("boiling water", True),
        ("green pepper", False),
        ("salt cod", False),
        ("sesame seed oil", False),
        ("hot beef stock", False),
    ],
)
def test_is_staple(name: str, staple: bool) -> None:
    """Only salt, pepper, water, oil and ice variants are staples."""
    assert is_staple(name) is staple


def test_every_substitute_is_recognised() -> None:
    """The checker can verify every replacement the substitution table offers."""
    replacements = {r for options in SUBSTITUTES.values() for r in options}
    assert [r for r in sorted(replacements) if not classify(r).known] == []


def test_no_substitute_keeps_all_tags_of_its_original() -> None:
    """A replacement always drops at least one property of what it replaces."""
    offenders = [
        (" ".join(original), replacement)
        for original, options in SUBSTITUTES.items()
        if (tags := classify(" ".join(original)).tags)
        for replacement in options
        if tags <= classify(replacement).tags
    ]
    assert offenders == []
