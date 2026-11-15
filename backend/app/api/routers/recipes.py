"""Recipe endpoints (`/api/recipes`), including semantic search."""

from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app import crud
from app.api.deps import DbSession, EmbedderDep
from app.core.exceptions import IngredientsNotFoundError
from app.models import Recipe
from app.schemas import RecipeCreate, RecipeRead, RecipeSearchRequest, RecipeSearchResponse
from app.services.indexing import try_index_recipe
from app.services.recipe_search import search_recipes

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.post("", response_model=RecipeRead, status_code=status.HTTP_201_CREATED)
async def create_recipe(payload: RecipeCreate, db: DbSession, embedder: EmbedderDep) -> Recipe:
    """Create a recipe together with its ingredient links, then index it for search.

    Indexing is best effort: if the embedding model is unavailable the recipe is still
    created, and `python -m app.cli reindex` indexes it later.

    Raises:
        HTTPException: 422 if any referenced ingredient id does not exist.
    """
    try:
        recipe_id = (await crud.recipe.create_recipe(db, payload)).id
    except IngredientsNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await try_index_recipe(db, recipe_id, embedder)
    # Re-load: a failed indexing attempt rolls back, which expires loaded instances.
    created = await crud.recipe.get_recipe(db, recipe_id)
    assert created is not None  # noqa: S101
    return created


@router.get("", response_model=list[RecipeRead])
async def list_recipes(
    db: DbSession,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> Sequence[Recipe]:
    """List recipes ordered by id, paginated with `skip` / `limit`."""
    return await crud.recipe.get_recipes(db, skip=skip, limit=limit)


@router.post("/search", response_model=RecipeSearchResponse)
async def search(
    payload: RecipeSearchRequest, db: DbSession, embedder: EmbedderDep
) -> RecipeSearchResponse:
    """Semantic search: the recipes whose chunks are most similar to the query.

    Returns 503 if the embedding model is unavailable.
    """
    return await search_recipes(db, payload, embedder)


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
