"""Aggregate router mounting all resource routers under `/api`."""

from fastapi import APIRouter

from app.api.routers import assistant, fridge, ingredients, profile, recipes

api_router = APIRouter(prefix="/api")
api_router.include_router(ingredients.router)
api_router.include_router(recipes.router)
api_router.include_router(assistant.router)
api_router.include_router(fridge.router)
api_router.include_router(profile.router)
