"""Liveness and dependency health endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_chat_client, get_embedding_client
from app.clients.ollama import OllamaClient, OllamaStatus

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Report that the API process is up."""
    return {"status": "ok"}


@router.get("/health/llm")
async def llm_health(
    chat: Annotated[OllamaClient, Depends(get_chat_client)],
    embeddings: Annotated[OllamaClient, Depends(get_embedding_client)],
) -> dict[str, OllamaStatus]:
    """Report whether the chat and embedding servers are reachable and have the models."""
    return {"chat": await chat.status(), "embeddings": await embeddings.status()}
