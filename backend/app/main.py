"""FastAPI application factory and the ASGI `app` (entrypoint `app.main:app`)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.api.routers import health
from app.clients.ollama import OllamaClient
from app.core.config import settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bootstrap the database and open the LLM clients for the app's lifetime."""
    await init_db()
    async with OllamaClient.for_chat() as chat, OllamaClient.for_embeddings() as embeddings:
        app.state.chat_client = chat
        app.state.embedding_client = embeddings
        yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Tests call this to get a fresh app whose dependencies can be overridden.
    """
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(api_router)
    return app


app = create_app()
