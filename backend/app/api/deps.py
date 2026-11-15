"""Reusable FastAPI dependencies for the routers.

The LLM dependencies are layered so tests can replace either the whole Ollama facade
(`get_ollama`) or just the models (`get_embedder`, `get_chat_model`) via
`app.dependency_overrides`.
"""

from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.llm import ChatModel, Embedder, Ollama

DbSession = Annotated[AsyncSession, Depends(get_db)]


def get_ollama(request: Request) -> Ollama:
    """Return the application-wide Ollama facade created in the lifespan handler."""
    return cast(Ollama, request.app.state.ollama)


OllamaDep = Annotated[Ollama, Depends(get_ollama)]


def get_embedder(ollama: OllamaDep) -> Embedder:
    """Return the embedding model used for indexing and retrieval."""
    return ollama.embedder


def get_chat_model(ollama: OllamaDep) -> ChatModel:
    """Return the chat model used to generate answers."""
    return ollama.chat_model


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
ChatModelDep = Annotated[ChatModel, Depends(get_chat_model)]
