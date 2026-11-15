"""Split recipes into retrieval chunks.

Strategy:

* one **overview** chunk per recipe: title, category, cuisine, tags and the ingredient
  list. This answers "what can I make with X" and "a quick Italian dinner" style queries;
* one or more **instructions** chunks: the method split on steps, packed greedily up to
  `max_chars` with the previous step repeated as overlap, so technique queries ("slow
  braise", "no oven") hit the relevant part of the method.

Every chunk starts with the recipe title so it stays meaningful when read on its own
(by the embedding model and by the LLM).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.models import Recipe

# "STEP 1", "Step 2:", "1.", "2)" at the start of a line in the source text.
_STEP_HEADER = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]?|\d+\s*[.)])\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkKind(StrEnum):
    """What part of a recipe a chunk covers."""

    OVERVIEW = "overview"
    INSTRUCTIONS = "instructions"


@dataclass(frozen=True, slots=True)
class RecipeDocument:
    """The text fields of a recipe that are chunked, decoupled from the ORM."""

    title: str
    instructions: str
    category: str | None = None
    area: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    ingredients: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> RecipeDocument:
        """Build a document from a recipe whose ingredients are loaded."""
        return cls(
            title=recipe.title,
            instructions=recipe.instructions,
            category=recipe.category,
            area=recipe.area,
            tags=tuple(recipe.tags or ()),
            ingredients=tuple(
                format_ingredient(item.ingredient.name, item.quantity, item.unit)
                for item in recipe.ingredients
            ),
        )


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk ready to be embedded."""

    kind: ChunkKind
    content: str


def format_ingredient(name: str, quantity: float | None, unit: str | None) -> str:
    """Render an ingredient line, e.g. `200 g pasta`, `pinch salt` or `garlic`."""
    amount = f"{quantity:g}" if quantity is not None else ""
    return " ".join(part for part in (amount, unit or "", name) if part)


def chunk_recipe(document: RecipeDocument, *, max_chars: int = 1000) -> list[ChunkDraft]:
    """Split a recipe into an overview chunk followed by instructions chunks.

    Args:
        document: The recipe text.
        max_chars: Soft upper bound for one instructions chunk (a single step longer than
            this is split on sentences; a single overlong sentence is kept whole).

    Returns:
        The chunks in reading order; never empty.
    """
    chunks = [ChunkDraft(ChunkKind.OVERVIEW, _overview(document))]

    windows = _pack(_steps(document.instructions), max_chars=max_chars)
    for index, window in enumerate(windows, start=1):
        part = f" (part {index}/{len(windows)})" if len(windows) > 1 else ""
        header = f"Recipe: {document.title}\nMethod{part}:\n"
        chunks.append(ChunkDraft(ChunkKind.INSTRUCTIONS, header + "\n".join(window)))
    return chunks


def _overview(document: RecipeDocument) -> str:
    """Title + metadata + ingredient list as one block of text."""
    lines = [f"Recipe: {document.title}"]
    if document.category:
        lines.append(f"Category: {document.category}")
    if document.area:
        lines.append(f"Cuisine: {document.area}")
    if document.tags:
        lines.append(f"Tags: {', '.join(document.tags)}")
    if document.ingredients:
        lines.append(f"Ingredients: {', '.join(document.ingredients)}")
    return "\n".join(lines)


def _steps(instructions: str) -> list[str]:
    """Split instructions into steps: one per non-empty line, without "STEP n" headers."""
    steps = []
    for line in instructions.splitlines():
        step = " ".join(_STEP_HEADER.sub("", line).split())
        if step:
            steps.append(step)
    return steps


def _pack(steps: list[str], *, max_chars: int) -> list[list[str]]:
    """Greedily pack steps into windows of at most `max_chars`, overlapping by one step."""
    pieces = [piece for step in steps for piece in _split_long(step, max_chars)]
    windows: list[list[str]] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) + 1 > max_chars:
            windows.append(current)
            # Carry the last piece over as context, unless it would overflow on its own.
            overlap = current[-1]
            current, size = (
                ([overlap], len(overlap)) if len(overlap) + len(piece) < max_chars else ([], 0)
            )
        current.append(piece)
        size += len(piece) + 1
    if current:
        windows.append(current)
    return windows


def _split_long(step: str, max_chars: int) -> list[str]:
    """Split a step longer than `max_chars` into sentence groups that fit."""
    if len(step) <= max_chars:
        return [step]
    groups: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(step):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups
