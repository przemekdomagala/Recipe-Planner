"""Tests for prompt construction and grounded answer generation."""

from app.rag import ChunkKind, RecipeHit, RetrievedChunk, generate_answer
from app.rag.generation import NO_CONTEXT_ANSWER, build_contexts
from app.rag.prompts import SYSTEM_PROMPT, RecipeContext, build_messages

from tests.fakes import FakeChatModel


def make_hit(recipe_id: int, title: str, *chunks: RetrievedChunk) -> RecipeHit:
    """Build a retrieval hit scored by its first chunk."""
    return RecipeHit(
        recipe_id=recipe_id,
        title=title,
        category=None,
        area=None,
        image_url=None,
        score=chunks[0].score if chunks else 0.0,
        chunks=chunks,
    )


def test_build_contexts_adds_summary_and_orders_instructions() -> None:
    """The summary comes first even if it did not match; instructions follow in order."""
    hit = make_hit(
        1,
        "Curry",
        RetrievedChunk(2, ChunkKind.INSTRUCTIONS, "part two", 0.9),
        RetrievedChunk(0, ChunkKind.SUMMARY, "summary (matched)", 0.8),
        RetrievedChunk(1, ChunkKind.INSTRUCTIONS, "part one", 0.7),
    )

    contexts = build_contexts([hit], {1: "summary"})

    assert contexts == [RecipeContext(title="Curry", passages=["summary", "part one", "part two"])]


def test_build_messages_puts_context_before_question() -> None:
    """The system prompt is sent first; the user prompt numbers the recipes."""
    messages = build_messages(
        "What is quick?",
        [RecipeContext("Toast", ["Recipe: Toast"]), RecipeContext("Soup", ["Recipe: Soup"])],
    )

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    user = messages[1].content
    assert user.index("[1] Toast") < user.index("[2] Soup") < user.index("Question: What is quick?")


async def test_generate_answer_calls_model_with_context() -> None:
    """The model receives the retrieved recipes and its reply is returned with the sources."""
    chat = FakeChatModel(reply="Make Curry.")
    hits = [make_hit(1, "Curry", RetrievedChunk(1, ChunkKind.INSTRUCTIONS, "Simmer.", 0.9))]

    answer = await generate_answer("Dinner idea?", hits, {1: "Recipe: Curry"}, chat)

    assert answer.text == "Make Curry."
    assert answer.model == "fake-chat"
    assert answer.sources == hits
    prompt = chat.received[0][1].content
    assert "Recipe: Curry" in prompt
    assert "Simmer." in prompt


async def test_generate_answer_without_hits_skips_the_model() -> None:
    """With nothing retrieved, a fixed answer is returned and no model is called."""
    chat = FakeChatModel()

    answer = await generate_answer("Anything?", [], {}, chat)

    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.model is None
    assert chat.received == []
