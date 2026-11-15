"""Recipe endpoints (`/api/recipes`), including semantic search."""

import logging
from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app import crud
from app.api.deps import DbSession, EmbedderDep
from app.api.mappers import to_search_hit
from app.core.exceptions import IngredientsNotFoundError, LLMServiceError
from app.models import Recipe
from app.rag import index_recipe, retrieve_recipes
from app.schemas import RecipeCreate, RecipeRead, RecipeSearchRequest, RecipeSearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.post("", response_model=RecipeRead, status_code=status.HTTP_201_CREATED)
async def create_recipe(payload: RecipeCreate, db: DbSession, embedder: EmbedderDep) -> Recipe:
    """Create a recipe together with its ingredient links and index it for search.

    Indexing is best-effort: if the embedding service is down, the recipe is still created
    and can be indexed later with `python -m app.cli reindex`.

    Raises:
        HTTPException: 422 if any referenced ingredient id does not exist.
    """
    try:
        recipe = await crud.recipe.create_recipe(db, payload)
    except IngredientsNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        await index_recipe(db, recipe, embedder)
    except LLMServiceError as exc:
        logger.warning("Recipe %s was created but not indexed: %s", recipe.id, exc)
    return recipe


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
    """Find the recipes semantically closest to a natural-language query.

    Raises:
        HTTPException: 503 if the embedding service is unavailable.
    """
    try:
        hits = await retrieve_recipes(
            db,
            embedder,
            payload.query,
            limit=payload.limit,
            category=payload.category,
            area=payload.area,
        )
    except LLMServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return RecipeSearchResponse(query=payload.query, results=[to_search_hit(hit) for hit in hits])


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
