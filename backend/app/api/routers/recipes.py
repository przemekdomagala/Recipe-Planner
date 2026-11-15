"""Recipe endpoints (`/api/recipes`), including the search placeholder."""

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app import crud
from app.api.deps import DbSession
from app.core.exceptions import IngredientsNotFoundError
from app.models import Recipe
from app.schemas import RecipeCreate, RecipeRead, RecipeSearchRequest, RecipeSearchResponse
from app.services.recipe_search import search_recipes

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.post("", response_model=RecipeRead, status_code=status.HTTP_201_CREATED)
async def create_recipe(payload: RecipeCreate, db: DbSession) -> Recipe:
    """Create a recipe together with its ingredient links.

    Raises:
        HTTPException: 422 if any referenced ingredient id does not exist.
    """
    try:
        return await crud.recipe.create_recipe(db, payload)
    except IngredientsNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[RecipeRead])
async def list_recipes(
    db: DbSession,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> Sequence[Recipe]:
    """List recipes ordered by id, paginated with `skip` / `limit`."""
    return await crud.recipe.get_recipes(db, skip=skip, limit=limit)


@router.post("/search", response_model=RecipeSearchResponse)
async def search(payload: RecipeSearchRequest, db: DbSession) -> RecipeSearchResponse:
    """Placeholder for RAG vector search; currently returns a mock response."""
    return await search_recipes(db, payload)


@router.get("/{recipe_id}", response_model=RecipeRead)
async def get_recipe(recipe_id: int, db: DbSession) -> Recipe:
    """Return a single recipe.

    Raises:
        HTTPException: 404 if the recipe does not exist.
    """
    recipe = await crud.recipe.get_recipe(db, recipe_id)
    if recipe is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipe not found")
    return recipe
