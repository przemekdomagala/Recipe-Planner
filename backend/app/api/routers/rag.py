"""RAG assistant endpoints (`/api/rag`): grounded question answering and readiness status."""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import ChatModelDep, DbSession, EmbedderDep, OllamaDep
from app.api.mappers import to_search_hit
from app.core.exceptions import LLMServiceError
from app.rag import answer_question, get_index_stats
from app.schemas import AskRequest, AskResponse, RagStatus

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest, db: DbSession, embedder: EmbedderDep, chat_model: ChatModelDep
) -> AskResponse:
    """Answer a question with the LLM, grounded on the most relevant recipes.

    Raises:
        HTTPException: 503 if the embedding or chat model is unavailable.
    """
    try:
        answer = await answer_question(
            db,
            payload.question,
            embedder=embedder,
            chat_model=chat_model,
            top_k=payload.top_k,
            category=payload.category,
            area=payload.area,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return AskResponse(
        question=payload.question,
        answer=answer.text,
        model=answer.model,
        sources=[to_search_hit(hit) for hit in answer.sources],
    )


@router.get("/status", response_model=RagStatus)
async def rag_status(db: DbSession, ollama: OllamaDep) -> RagStatus:
    """Report how many recipes are indexed and whether Ollama serves the configured models."""
    stats = await get_index_stats(db)
    detail: str | None = None
    try:
        missing = await ollama.missing_models()
    except LLMServiceError as exc:
        missing = [ollama.embedder.model_name, ollama.chat_model.model_name]
        detail = str(exc)
    else:
        if missing:
            detail = f"Models not pulled yet: {', '.join(missing)}"
    return RagStatus(
        total_recipes=stats.total_recipes,
        indexed_recipes=stats.indexed_recipes,
        chunks=stats.chunks,
        embedding_model=ollama.embedder.model_name,
        chat_model=ollama.chat_model.model_name,
        llm_available=detail is None,
        missing_models=missing,
        detail=detail,
    )
