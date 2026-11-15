"""Recipe assistant endpoints (`/api/assistant`): RAG answers from the chat model."""

from fastapi import APIRouter

from app.api.deps import ChatModelDep, DbSession, EmbedderDep
from app.schemas import AskRequest, AskResponse
from app.services.assistant import answer_question

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest, db: DbSession, embedder: EmbedderDep, chat_model: ChatModelDep
) -> AskResponse:
    """Answer a cooking question grounded in the recipes retrieved for it.

    Returns 503 if the LLM backend is unreachable, unauthorized or lacks the model.
    """
    return await answer_question(db, payload, embedder, chat_model)
