"""FastAPI application factory and the ASGI `app` (entrypoint `app.main:app`)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.routers import health
from app.core.config import settings
from app.db.session import init_db
from app.llm import ollama_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Bootstrap the database and open the Ollama client for the app's lifetime.

    The client is stored on `app.state.ollama` and read by `app.api.deps.get_ollama`.
    """
    await init_db()
    async with ollama_client(settings) as ollama:
        app.state.ollama = ollama
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
    app.include_router(health.router)
    app.include_router(api_router)
    return app


app = create_app()
