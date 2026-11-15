"""Splitting recipes into retrievable text chunks (pure functions, no I/O)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import Recipe, RecipeIngredient

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkKind(StrEnum):
    """What part of a recipe a chunk was built from."""

    SUMMARY = "summary"
    INSTRUCTIONS = "instructions"


@dataclass(frozen=True, slots=True)
class RecipeDocument:
    """The text of a recipe that gets indexed, decoupled from the ORM models."""

    title: str
    instructions: str
    category: str | None = None
    area: str | None = None
    tags: Sequence[str] = ()
    ingredients: Sequence[str] = ()

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> RecipeDocument:
        """Build the document from a recipe whose ingredients are already loaded.

        Args:
            recipe: Recipe with `ingredients` (and their `ingredient`) eagerly loaded.
        """
        return cls(
            title=recipe.title,
            instructions=recipe.instructions,
            category=recipe.category,
            area=recipe.area,
            tags=tuple(recipe.tags or ()),
            ingredients=tuple(format_ingredient_line(item) for item in recipe.ingredients),
        )


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A chunk ready to be embedded; `index` is its position within the recipe."""

    index: int
    kind: ChunkKind
    content: str


@dataclass(frozen=True, slots=True)
class _Sentence:
    """A sentence (or a piece of an over-long one) and the paragraph it belongs to."""

    paragraph: int
    text: str


def format_ingredient_line(item: RecipeIngredient) -> str:
    """Render an ingredient line, e.g. "garlic (2 cloves minced)" or "pasta (200 g)".

    The original free-text measure is preferred because it carries preparation details;
    otherwise the parsed quantity and unit are used.
    """
    name = item.ingredient.name
    if item.measure:
        amount = item.measure
    else:
        quantity = f"{item.quantity:g}" if item.quantity is not None else ""
        amount = " ".join(part for part in (quantity, item.unit or "") if part)
    return f"{name} ({amount})" if amount else name


def build_summary(document: RecipeDocument) -> str:
    """Return the summary chunk text: title, metadata and the full ingredient list.

    The summary is never split, so a question like "what can I cook with chicken and
    rice?" matches one chunk that holds all of a recipe's ingredients.
    """
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


def _split_words(text: str, max_chars: int) -> list[str]:
    """Greedily pack the words of `text` into pieces of at most `max_chars` characters."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _split_sentences(text: str, max_chars: int) -> list[_Sentence]:
    """Split text into sentences tagged with their paragraph; over-long ones are cut by words."""
    sentences: list[_Sentence] = []
    paragraphs = (p for p in _PARAGRAPH_BREAK.split(text) if p.strip())
    for paragraph_no, paragraph in enumerate(paragraphs):
        for sentence in _SENTENCE_END.split(" ".join(paragraph.split())):
            sentences.extend(
                _Sentence(paragraph_no, piece) for piece in _split_words(sentence, max_chars)
            )
    return sentences


def _join(sentences: Sequence[_Sentence]) -> str:
    """Join sentences with a space inside a paragraph and a newline between paragraphs."""
    parts: list[str] = []
    for i, sentence in enumerate(sentences):
        if i:
            parts.append(" " if sentence.paragraph == sentences[i - 1].paragraph else "\n")
        parts.append(sentence.text)
    return "".join(parts)


def _overlap_tail(sentences: Sequence[_Sentence], overlap_chars: int) -> list[_Sentence]:
    """Return the longest run of trailing sentences that fits in `overlap_chars`."""
    tail: list[_Sentence] = []
    for sentence in reversed(sentences):
        if len(_join([sentence, *tail])) > overlap_chars:
            break
        tail.insert(0, sentence)
    return tail


def split_text(text: str, *, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split text into chunks of at most `max_chars`, cutting at sentence boundaries.

    Sentences are packed greedily. Each new chunk starts with the trailing sentences of the
    previous one (up to `overlap_chars`), so a step that spans a boundary keeps its context.
    Sentences longer than `max_chars` are cut at word boundaries.

    Args:
        text: Text with paragraphs separated by blank lines.
        max_chars: Maximum chunk length.
        overlap_chars: Maximum length of the overlap carried into the next chunk.

    Returns:
        The chunks in order; empty if `text` is blank.

    Raises:
        ValueError: If `max_chars` is not positive or `overlap_chars` is not in
            `[0, max_chars)`.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    chunks: list[str] = []
    current: list[_Sentence] = []
    for sentence in _split_sentences(text, max_chars):
        if current and len(_join([*current, sentence])) > max_chars:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap_chars)
            while current and len(_join([*current, sentence])) > max_chars:
                current.pop(0)
        current.append(sentence)
    if current:
        chunks.append(_join(current))
    return chunks


def chunk_recipe(
    document: RecipeDocument, *, max_chars: int, overlap_chars: int = 0
) -> list[TextChunk]:
    """Split a recipe into one summary chunk followed by its instruction chunks.

    Every instruction chunk starts with a header naming the recipe, so it stays
    meaningful when retrieved on its own. `max_chars` limits the instruction text of a
    chunk, not the header or the summary.

    Args:
        document: The recipe text to index.
        max_chars: Maximum length of the instruction text per chunk.
        overlap_chars: Overlap between consecutive instruction chunks.
    """
    chunks = [TextChunk(index=0, kind=ChunkKind.SUMMARY, content=build_summary(document))]
    parts = split_text(document.instructions, max_chars=max_chars, overlap_chars=overlap_chars)
    for part_no, part in enumerate(parts, start=1):
        header = f"Recipe: {document.title} - preparation (part {part_no} of {len(parts)})"
        chunks.append(
            TextChunk(index=part_no, kind=ChunkKind.INSTRUCTIONS, content=f"{header}\n{part}")
        )
    return chunks
