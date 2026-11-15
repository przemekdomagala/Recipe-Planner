"""Tests for splitting recipes into retrieval chunks."""

import itertools

from app.services.chunking import ChunkKind, RecipeDocument, chunk_recipe, format_ingredient


def test_overview_chunk_contains_metadata_and_ingredients() -> None:
    """The first chunk summarizes title, category, cuisine, tags and ingredients."""
    document = RecipeDocument(
        title="Teriyaki Chicken",
        instructions="Bake.",
        category="Chicken",
        area="Japanese",
        tags=("Meat", "Casserole"),
        ingredients=("3 tbsp soy sauce", "chicken breasts"),
    )

    overview = chunk_recipe(document)[0]

    assert overview.kind is ChunkKind.OVERVIEW
    assert overview.content == (
        "Recipe: Teriyaki Chicken\nCategory: Chicken\nCuisine: Japanese\n"
        "Tags: Meat, Casserole\nIngredients: 3 tbsp soy sauce, chicken breasts"
    )


def test_short_instructions_give_one_chunk_without_step_headers() -> None:
    """Steps are split on lines and "STEP n" / "1." headers are removed."""
    document = RecipeDocument(title="Toast", instructions="STEP 1\r\nToast bread.\n\n2. Butter it.")

    chunks = chunk_recipe(document)

    assert [chunk.kind for chunk in chunks] == [ChunkKind.OVERVIEW, ChunkKind.INSTRUCTIONS]
    assert chunks[1].content == "Recipe: Toast\nMethod:\nToast bread.\nButter it."


def test_long_instructions_are_windowed_with_overlap() -> None:
    """Windows stay within the size limit, are numbered and repeat the previous step."""
    steps = [f"Step number {i} " + "stir " * 30 for i in range(8)]
    document = RecipeDocument(title="Risotto", instructions="\n".join(steps))

    windows = chunk_recipe(document, max_chars=400)[1:]

    assert len(windows) > 1
    assert windows[0].content.startswith(f"Recipe: Risotto\nMethod (part 1/{len(windows)}):\n")
    bodies = [window.content.split(":\n", 1)[1].splitlines() for window in windows]
    assert all(len("\n".join(body)) <= 400 for body in bodies)
    assert all(prev[-1] == nxt[0] for prev, nxt in itertools.pairwise(bodies))
    assert {line for body in bodies for line in body} == {step.strip() for step in steps}


def test_overlong_step_is_split_on_sentences() -> None:
    """A single step longer than the limit is cut at sentence boundaries."""
    step = " ".join(f"Sentence {i} is here." for i in range(60))
    windows = chunk_recipe(RecipeDocument(title="X", instructions=step), max_chars=300)[1:]

    assert len(windows) > 1
    assert all(line.endswith(".") for w in windows for line in w.content.splitlines()[2:])


def test_format_ingredient() -> None:
    """Amounts render compactly and missing parts are omitted."""
    assert format_ingredient("pasta", 200.0, "g") == "200 g pasta"
    assert format_ingredient("soy sauce", 0.75, "cup") == "0.75 cup soy sauce"
    assert format_ingredient("salt", None, "pinch") == "pinch salt"
    assert format_ingredient("garlic", None, None) == "garlic"
