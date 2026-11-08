from fastapi import APIRouter

from app.api.routers import ingredients, recipes

api_router = APIRouter(prefix="/api")
api_router.include_router(ingredients.router)
api_router.include_router(recipes.router)
