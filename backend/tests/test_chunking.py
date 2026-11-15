"""Tests for splitting recipes into chunks."""

import itertools

import pytest
from app.rag import ChunkKind, RecipeDocument, chunk_recipe, split_text


def test_split_text_keeps_short_text_in_one_chunk() -> None:
    """Text shorter than the limit is returned unchanged (whitespace normalized)."""
    assert split_text("Boil water.  Add pasta.", max_chars=100) == ["Boil water. Add pasta."]


def test_split_text_respects_limit_and_sentence_boundaries() -> None:
    """Chunks never exceed the limit and are cut between sentences."""
    text = " ".join(f"Step number {i} is done." for i in range(20))

    chunks = split_text(text, max_chars=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)
    assert " ".join(chunks) == text


def test_split_text_overlaps_consecutive_chunks() -> None:
    """Each chunk starts with the last sentence of the previous one when it fits."""
    sentences = [f"Sentence {i} here." for i in range(6)]

    chunks = split_text(" ".join(sentences), max_chars=60, overlap_chars=20)

    assert len(chunks) > 1
    for previous, current in itertools.pairwise(chunks):
        last_sentence = previous.rsplit(". ", 1)[-1]
        assert current.startswith(last_sentence.rstrip("."))


def test_split_text_preserves_paragraph_breaks() -> None:
    """Sentences of different paragraphs are joined with a newline."""
    assert split_text("Mix.\n\nBake.", max_chars=100) == ["Mix.\nBake."]


def test_split_text_cuts_overlong_sentences_by_words() -> None:
    """A sentence longer than the limit is split at word boundaries."""
    chunks = split_text("word " * 50, max_chars=24)

    assert all(len(chunk) <= 24 for chunk in chunks)
    assert sum(chunk.count("word") for chunk in chunks) == 50


def test_split_text_blank_input() -> None:
    """Blank text yields no chunks."""
    assert split_text("  \n\n ", max_chars=10) == []


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_split_text_validates_parameters(max_chars: int, overlap: int) -> None:
    """Non-positive limits and overlaps not smaller than the limit are rejected."""
    with pytest.raises(ValueError):
        split_text("text", max_chars=max_chars, overlap_chars=overlap)


def test_chunk_recipe_builds_summary_and_instruction_chunks() -> None:
    """The first chunk summarizes the recipe; instruction chunks carry a header."""
    document = RecipeDocument(
        title="Pancakes",
        instructions="Whisk flour, eggs and milk. Rest the batter.\n\nFry thin pancakes.",
        category="Dessert",
        area="French",
        tags=("Sweet", "Breakfast"),
        ingredients=("flour (100 g)", "eggs (2)", "milk (300ml)"),
    )

    chunks = chunk_recipe(document, max_chars=50)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    summary = chunks[0]
    assert summary.kind is ChunkKind.SUMMARY
    assert summary.content == (
        "Recipe: Pancakes\n"
        "Category: Dessert\n"
        "Cuisine: French\n"
        "Tags: Sweet, Breakfast\n"
        "Ingredients: flour (100 g), eggs (2), milk (300ml)"
    )
    instructions = chunks[1:]
    assert len(instructions) == 2
    assert all(chunk.kind is ChunkKind.INSTRUCTIONS for chunk in instructions)
    assert instructions[0].content.startswith("Recipe: Pancakes - preparation (part 1 of 2)\n")
    assert "Fry thin pancakes." in instructions[1].content


def test_chunk_recipe_summary_omits_missing_metadata() -> None:
    """Absent metadata lines are left out of the summary."""
    chunks = chunk_recipe(RecipeDocument(title="Water", instructions="Pour."), max_chars=100)

    assert chunks[0].content == "Recipe: Water"
    assert chunks[1].content.endswith("\nPour.")
