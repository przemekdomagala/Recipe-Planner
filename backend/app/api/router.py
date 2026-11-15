"""Aggregate router mounting all resource routers under `/api`."""

from fastapi import APIRouter

from app.api.routers import ingredients, rag, recipes

api_router = APIRouter(prefix="/api")
api_router.include_router(ingredients.router)
api_router.include_router(recipes.router)
api_router.include_router(rag.router)
