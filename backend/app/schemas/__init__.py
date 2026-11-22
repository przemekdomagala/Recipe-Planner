"""Pydantic request/response schemas for the API."""

from app.schemas.assistant import AskRequest, AskResponse
from app.schemas.fridge import (
    FridgeItemCreate,
    FridgeItemRead,
    FridgeItemUpdate,
    FridgeSuggestion,
    Substitution,
    SuggestionRequest,
    SuggestionResponse,
)
from app.schemas.ingest import ExternalIngredient, ExternalRecipe
from app.schemas.ingredient import IngredientCreate, IngredientRead, normalize_ingredient_name
from app.schemas.profile import DietOption, DietProfileData, LabeledOption, ProfileOptions
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
    "DietOption",
    "DietProfileData",
    "ExternalIngredient",
    "ExternalRecipe",
    "FridgeItemCreate",
    "FridgeItemRead",
    "FridgeItemUpdate",
    "FridgeSuggestion",
    "IngredientCreate",
    "IngredientRead",
    "LabeledOption",
    "MatchedChunk",
    "ProfileOptions",
    "RecipeCreate",
    "RecipeIngredientCreate",
    "RecipeIngredientRead",
    "RecipeRead",
    "RecipeSearchFilters",
    "RecipeSearchHit",
    "RecipeSearchRequest",
    "RecipeSearchResponse",
    "Substitution",
    "SuggestionRequest",
    "SuggestionResponse",
    "normalize_ingredient_name",
]
