"""Reusable FastAPI dependencies for the routers."""

from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.ollama import OllamaClient
from app.db.session import get_db
from app.services.llm import ChatModel, Embedder

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_chat_client(request: Request) -> OllamaClient:
    """The chat-model client created in the app lifespan."""
    return cast(OllamaClient, request.app.state.chat_client)


def get_embedding_client(request: Request) -> OllamaClient:
    """The embedding-model client created in the app lifespan."""
    return cast(OllamaClient, request.app.state.embedding_client)


def get_embedder(client: Annotated[OllamaClient, Depends(get_embedding_client)]) -> Embedder:
    """The embedding model; tests override this with a deterministic fake."""
    return client


def get_chat_model(client: Annotated[OllamaClient, Depends(get_chat_client)]) -> ChatModel:
    """The chat model; tests override this with a fake."""
    return client


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ChatModelDep = Annotated[ChatModel, Depends(get_chat_model)]
