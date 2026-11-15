"""Retrieval-augmented answer generation: retrieve recipes, then ask the chat model."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import ChatModel, Embedder
from app.rag.chunking import ChunkKind
from app.rag.prompts import RecipeContext, build_messages
from app.rag.retrieval import RecipeHit, fetch_summary_chunks, retrieve_recipes

NO_CONTEXT_ANSWER = (
    "I couldn't find any matching recipes in the knowledge base. "
    "Try rephrasing the question or relaxing the filters, or import recipes first."
)


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer and the recipes it was grounded on."""

    text: str
    sources: Sequence[RecipeHit]
    # `None` when no model was called because nothing relevant was retrieved.
    model: str | None


def build_contexts(hits: Sequence[RecipeHit], summaries: Mapping[int, str]) -> list[RecipeContext]:
    """Combine each hit's summary chunk with its matched instruction chunks.

    The summary (ingredients and metadata) is always included, even if it did not match,
    so the model knows what a recipe needs; instruction chunks follow in recipe order.

    Args:
        hits: Retrieved recipes in ranking order.
        summaries: Summary chunk text per recipe id.
    """
    contexts: list[RecipeContext] = []
    for hit in hits:
        passages: list[str] = []
        if summary := summaries.get(hit.recipe_id):
            passages.append(summary)
        instructions = sorted(
            (chunk for chunk in hit.chunks if chunk.kind is ChunkKind.INSTRUCTIONS),
            key=lambda chunk: chunk.chunk_index,
        )
        passages.extend(chunk.content for chunk in instructions)
        contexts.append(RecipeContext(title=hit.title, passages=passages))
    return contexts


async def generate_answer(
    question: str, hits: Sequence[RecipeHit], summaries: Mapping[int, str], chat_model: ChatModel
) -> Answer:
    """Answer `question` from already retrieved recipes.

    The chat model is not called when nothing was retrieved: an ungrounded answer would
    defeat the purpose of RAG, so a fixed message is returned instead.

    Args:
        question: The user's question.
        hits: Retrieved recipes in ranking order.
        summaries: Summary chunk text per recipe id.
        chat_model: Model generating the answer.

    Raises:
        LLMServiceError: If the chat model fails.
    """
    if not hits:
        return Answer(text=NO_CONTEXT_ANSWER, sources=[], model=None)
    messages = build_messages(question, build_contexts(hits, summaries))
    reply = await chat_model.chat(messages)
    return Answer(text=reply, sources=hits, model=chat_model.model_name)


async def answer_question(
    db: AsyncSession,
    question: str,
    *,
    embedder: Embedder,
    chat_model: ChatModel,
    top_k: int,
    category: str | None = None,
    area: str | None = None,
) -> Answer:
    """Run the full RAG pipeline: retrieve the `top_k` best recipes and generate an answer.

    Args:
        db: Database session.
        question: The user's question.
        embedder: Embedding model used for retrieval.
        chat_model: Model generating the answer.
        top_k: Number of recipes given to the model as context.
        category: Restrict retrieval to this recipe category.
        area: Restrict retrieval to this cuisine/area.

    Raises:
        LLMServiceError: If embedding or generation fails.
    """
    hits = await retrieve_recipes(db, embedder, question, limit=top_k, category=category, area=area)
    summaries = await fetch_summary_chunks(db, [hit.recipe_id for hit in hits])
    return await generate_answer(question, hits, summaries, chat_model)
