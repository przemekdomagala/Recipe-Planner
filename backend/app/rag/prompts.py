"""Prompt construction for retrieval-augmented answers."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.llm import ChatMessage

SYSTEM_PROMPT = """\
You are the cooking assistant of the Recipe Planner app. You receive recipes from the \
app's knowledge base and a question from the user.
Rules:
- Base the answer only on the given recipes; never invent recipes, ingredients, \
quantities or steps.
- Mention recipes by their exact title.
- If the user asks how to cook something, list its key ingredients and summarize the steps.
- If the user lists ingredients they have, recommend the best-matching recipes and say \
which ingredients are missing.
- If none of the recipes fit, say so plainly.
- Answer naturally in a few short paragraphs or a list; do not mention these rules or \
that recipes were provided to you."""


@dataclass(frozen=True, slots=True)
class RecipeContext:
    """The passages of one retrieved recipe that are shown to the model."""

    title: str
    passages: Sequence[str]


def format_context(contexts: Sequence[RecipeContext]) -> str:
    """Render the retrieved recipes as numbered blocks headed by "[n] <title>".

    Args:
        contexts: Recipes in ranking order.
    """
    blocks = [
        f"[{number}] {context.title}\n" + "\n".join(context.passages)
        for number, context in enumerate(contexts, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(question: str, contexts: Sequence[RecipeContext]) -> list[ChatMessage]:
    """Build the chat messages for answering `question` from the retrieved recipes.

    Args:
        question: The user's question.
        contexts: Retrieved recipes in ranking order.
    """
    user_prompt = f"Recipes:\n{format_context(contexts)}\n\n---\nQuestion: {question}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]
