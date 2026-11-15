"""Parsing of free-text ingredient measures ("1 1/2 cups", "100g", "Pinch") into amounts."""

import re
from dataclasses import dataclass

_VULGAR_FRACTIONS = {
    "½": 1 / 2,
    "⅓": 1 / 3,
    "⅔": 2 / 3,
    "¼": 1 / 4,
    "¾": 3 / 4,
    "⅕": 1 / 5,
    "⅛": 1 / 8,
}
_VULGAR = "".join(_VULGAR_FRACTIONS)

_MIXED = re.compile(r"(\d+)\s+(\d+)/(\d+)")
_FRACTION = re.compile(r"(\d+)/(\d+)")
_DECIMAL = re.compile(rf"(\d+(?:[.,]\d+)?)?\s*([{_VULGAR}])?")

# Canonical unit -> spellings found in recipe data (all lowercase, without trailing dots).
_UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "g": ("g", "gr", "gram", "grams", "gramme", "grammes"),
    "kg": ("kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"),
    "ml": ("ml", "millilitre", "millilitres", "milliliter", "milliliters"),
    "l": ("l", "ltr", "litre", "litres", "liter", "liters"),
    "tsp": ("tsp", "tsps", "teaspoon", "teaspoons"),
    "tbsp": ("tbsp", "tbsps", "tbs", "tbls", "tblsp", "tblsps", "tablespoon", "tablespoons"),
    "cup": ("cup", "cups"),
    "fl oz": ("fl oz", "fl. oz", "fluid ounce", "fluid ounces"),
    "oz": ("oz", "ounce", "ounces"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "pint": ("pint", "pints"),
    "quart": ("quart", "quarts", "qt"),
    "pinch": ("pinch", "pinches"),
    "dash": ("dash", "dashes"),
    "splash": ("splash", "splashes"),
    "handful": ("handful", "handfuls"),
    "bunch": ("bunch", "bunches"),
    "sprig": ("sprig", "sprigs"),
    "clove": ("clove", "cloves"),
    "slice": ("slice", "slices"),
    "can": ("can", "cans", "tin", "tins"),
    "stick": ("stick", "sticks"),
}
_UNIT_LOOKUP = {
    spelling: unit for unit, spellings in _UNIT_SPELLINGS.items() for spelling in spellings
}
# Two-word spellings ("fl oz") are tried before single words.
_MULTI_WORD_SPELLINGS = sorted((s for s in _UNIT_LOOKUP if " " in s), key=len, reverse=True)
# Units that imply a quantity of one when the measure has no number ("Pinch" = 1 pinch).
_COUNTABLE_WITHOUT_NUMBER = {"pinch", "dash", "splash", "handful", "bunch", "sprig"}


@dataclass(frozen=True, slots=True)
class ParsedMeasure:
    """Machine-readable part of a measure; either field is `None` when not recognized."""

    quantity: float | None
    unit: str | None


def _parse_quantity(text: str) -> tuple[float | None, str]:
    """Split a leading quantity off `text`.

    Supports integers, decimals (with "." or ","), fractions ("1/2"), mixed numbers
    ("1 1/2") and Unicode vulgar fractions ("½", "1½"). For ranges ("2-3") the lower bound
    is used.

    Returns:
        The quantity (or `None` if `text` does not start with one) and the remaining text.
    """
    if match := _MIXED.match(text):
        whole, numerator, denominator = (int(group) for group in match.groups())
        if denominator:
            return whole + numerator / denominator, text[match.end() :]
    if match := _FRACTION.match(text):
        numerator, denominator = int(match[1]), int(match[2])
        if denominator:
            return numerator / denominator, text[match.end() :]
    match = _DECIMAL.match(text)
    if match and (match[1] or match[2]):
        number = float(match[1].replace(",", ".")) if match[1] else 0.0
        fraction = _VULGAR_FRACTIONS[match[2]] if match[2] else 0.0
        return number + fraction, text[match.end() :]
    return None, text


def _parse_unit(text: str) -> str | None:
    """Return the canonical unit `text` starts with, or `None`."""
    lowered = text.lower()
    for spelling in _MULTI_WORD_SPELLINGS:
        if lowered.startswith(spelling) and lowered[len(spelling) : len(spelling) + 1] in ("", " "):
            return _UNIT_LOOKUP[spelling]
    first_word = re.split(r"[\s(,]", lowered, maxsplit=1)[0].rstrip(".")
    return _UNIT_LOOKUP.get(first_word)


def parse_measure(measure: str | None) -> ParsedMeasure:
    """Extract a quantity and a canonical unit from a free-text measure.

    Examples:
        "1 1/2 cups (360 milliliters)" -> 1.5 cup; "100g" -> 100 g; "2 tblsp" -> 2 tbsp;
        "Pinch" -> 1 pinch; "2 chopped" -> 2 (no unit); "To taste" -> nothing.

    Args:
        measure: Raw measure text, possibly empty or `None`.
    """
    text = " ".join((measure or "").split())
    if not text:
        return ParsedMeasure(quantity=None, unit=None)
    quantity, rest = _parse_quantity(text)
    if quantity is not None:
        # Range such as "2-3 tbsp": skip the upper bound.
        rest = re.sub(r"^\s*-\s*\d+(?:[.,]\d+)?", "", rest)
    unit = _parse_unit(rest.strip())
    if quantity is None and unit in _COUNTABLE_WITHOUT_NUMBER:
        quantity = 1.0
    if quantity is not None and quantity <= 0:
        quantity = None
    return ParsedMeasure(quantity=quantity, unit=unit)
