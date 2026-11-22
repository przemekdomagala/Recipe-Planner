"""Token-level matching of free-text ingredient names.

Ingredient names come from users, TheMealDB and the LLM, so they are compared on
normalized tokens rather than as raw strings: accents are stripped, words are lowercased
and naively singularized, and a phrase "matches" a name when its tokens appear there
contiguously. So "egg" matches "Egg Yolks" but not "Eggplant", and "chicken" matches
"chicken breast".
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence

_WORD = re.compile(r"[a-z]+")

Tokens = tuple[str, ...]


def _singular(word: str) -> str:
    """Naive English singular: good enough to match "tomatoes"/"tomato" or "berries"/"berry"."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> Tokens:
    """Split text into accent-free, lowercase, singular word tokens (single letters dropped)."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return tuple(_singular(word) for word in _WORD.findall(ascii_text.lower()) if len(word) > 1)


def contains_phrase(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    """Whether `phrase` occurs in `tokens` as a contiguous run (an empty phrase never does)."""
    size = len(phrase)
    if size == 0:
        return False
    return any(tuple(tokens[i : i + size]) == tuple(phrase) for i in range(len(tokens) - size + 1))


def mentions(name: str, phrase: str) -> bool:
    """Whether the ingredient `name` contains `phrase`, e.g. "Chicken Breast" and "chicken"."""
    return contains_phrase(tokenize(name), tokenize(phrase))


def same_ingredient(first: str, second: str) -> bool:
    """Loose equivalence: either name contains the other ("chicken" ~ "chicken thighs")."""
    return mentions(first, second) or mentions(second, first)


def find_ingredient(names: Iterable[str], wanted: str) -> str | None:
    """Return the first of `names` that is the same ingredient as `wanted`, if any."""
    return next((name for name in names if same_ingredient(name, wanted)), None)
