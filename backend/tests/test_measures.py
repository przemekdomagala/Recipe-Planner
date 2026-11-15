"""Tests for parsing free-text ingredient measures."""

import pytest
from app.sources import ParsedMeasure, parse_measure


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("100g", ParsedMeasure(100.0, "g")),
        ("2kg cut into 3cm cubes", ParsedMeasure(2.0, "kg")),
        ("1 tsp", ParsedMeasure(1.0, "tsp")),
        ("2 tblsp", ParsedMeasure(2.0, "tbsp")),
        ("3  tablespoons", ParsedMeasure(3.0, "tbsp")),
        ("1/2 cup", ParsedMeasure(0.5, "cup")),
        ("1 1/2 cups (360 milliliters)", ParsedMeasure(1.5, "cup")),
        ("½ tsp dissolved in ½ cup warm milk", ParsedMeasure(0.5, "tsp")),
        ("1½ lbs", ParsedMeasure(1.5, "lb")),
        ("2,5 l", ParsedMeasure(2.5, "l")),
        ("2-3 cloves", ParsedMeasure(2.0, "clove")),
        ("4 fl oz", ParsedMeasure(4.0, "fl oz")),
        ("2 chopped", ParsedMeasure(2.0, None)),
        ("Pinch", ParsedMeasure(1.0, "pinch")),
        ("To taste", ParsedMeasure(None, None)),
        ("Juice of 2", ParsedMeasure(None, None)),
        ("", ParsedMeasure(None, None)),
        (None, ParsedMeasure(None, None)),
        ("0 g", ParsedMeasure(None, "g")),
        ("1/0 cup", ParsedMeasure(1.0, None)),
    ],
)
def test_parse_measure(measure: str | None, expected: ParsedMeasure) -> None:
    """Quantities and canonical units are extracted from common measure spellings."""
    assert parse_measure(measure) == expected
