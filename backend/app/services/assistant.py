"""Recipe assistant: retrieval-augmented answers from the chat model.

Flow: retrieve the most relevant recipes (`recipe_search`), put the full recipes into the
prompt as numbered context, and let the chat model answer using only that context.
"""

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.models import Recipe
from app.schemas import AskRequest, AskResponse, RecipeSearchRequest
from app.services.chunking import format_ingredient
from app.services.llm import ChatMessage, ChatModel, Embedder
from app.services.recipe_search import search_recipes

NO_MATCH_ANSWER = (
    "I couldn't find any matching recipes in the knowledge base. "
    "Try rephrasing the question or loosening the filters."
)
# Keeps the prompt short: on CPU, prompt processing dominates the time to a new answer.
_MAX_INSTRUCTION_CHARS = 800

SYSTEM_PROMPT = """\
You are the cooking assistant of a meal-planning app.
Answer the user's question using ONLY the recipes provided in the context.
- Refer to recipes by their exact title and cite them like [1], [2] using the context numbers.
- If none of the recipes fits, say so plainly instead of inventing a recipe.
- Be brief: at most 120 words. Recommend the best-fitting recipes in 1-2 sentences each,
  with key ingredients and timings. Don't repeat whole recipes.
- Answer in the same language as the question."""


def build_context(recipes: Sequence[Recipe]) -> str:
    """Render recipes (ingredients loaded) as numbered context blocks for the prompt."""
    blocks = []
    for number, recipe in enumerate(recipes, start=1):
        meta = ", ".join(part for part in (recipe.category, recipe.area) if part)
        ingredients = "\n".join(
            f"- {format_ingredient(item.ingredient.name, item.quantity, item.unit)}"
            for item in recipe.ingredients
        )
        instructions = recipe.instructions
        if len(instructions) > _MAX_INSTRUCTION_CHARS:
            instructions = instructions[:_MAX_INSTRUCTION_CHARS].rstrip() + " [...]"
        blocks.append(
            f"[{number}] {recipe.title}" + (f" ({meta})" if meta else "") + "\n"
            f"Ingredients:\n{ingredients or '- (not listed)'}\n"
            f"Instructions:\n{instructions}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, recipes: Sequence[Recipe]) -> list[ChatMessage]:
    """Assemble the system prompt, the retrieved context and the user's question."""
    user_prompt = f"Context recipes:\n\n{build_context(recipes)}\n\nQuestion: {question}"
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user_prompt)]


async def answer_question(
    db: AsyncSession, request: AskRequest, embedder: Embedder, chat_model: ChatModel
) -> AskResponse:
    """Retrieve recipes relevant to the question and ask the chat model to answer.

    If nothing is retrieved the LLM is not called and a fixed answer is returned.

    Raises:
        LLMError: If embedding or generation fails.
    """
    search = await search_recipes(
        db,
        RecipeSearchRequest(
            query=request.question, limit=request.max_recipes, filters=request.filters
        ),
        embedder,
    )
    if not search.results:
        return AskResponse(
            question=request.question, answer=NO_MATCH_ANSWER, model=None, sources=[]
        )

    recipes = await crud.recipe.get_recipes_by_ids(db, [hit.recipe_id for hit in search.results])
    answer = await chat_model.chat(build_messages(request.question, recipes))
    return AskResponse(
        question=request.question,
        answer=answer,
        model=chat_model.chat_model,
        sources=search.results,
    )
