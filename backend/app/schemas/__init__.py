"""Pydantic request/response schemas for the API."""

from app.schemas.assistant import AskRequest, AskResponse
from app.schemas.ingest import ExternalIngredient, ExternalRecipe
from app.schemas.ingredient import IngredientCreate, IngredientRead, normalize_ingredient_name
from app.schemas.recipe import (
    RecipeCreate,
    RecipeIngredientCreate,
    RecipeIngredientRead,
    RecipeRead,
)
from app.schemas.search import (
    MatchedChunk,
    RecipeSearchFilters,
    RecipeSearchHit,
    RecipeSearchRequest,
    RecipeSearchResponse,
)

__all__ = [
    "AskRequest",
    "AskResponse",
    "ExternalIngredient",
    "ExternalRecipe",
    "IngredientCreate",
    "IngredientRead",
    "MatchedChunk",
    "RecipeCreate",
    "RecipeIngredientCreate",
    "RecipeIngredientRead",
    "RecipeRead",
    "RecipeSearchFilters",
    "RecipeSearchHit",
    "RecipeSearchRequest",
    "RecipeSearchResponse",
    "normalize_ingredient_name",
]
