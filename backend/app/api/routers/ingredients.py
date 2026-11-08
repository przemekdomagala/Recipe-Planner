from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app import crud
from app.api.deps import DbSession
from app.core.exceptions import DuplicateIngredientError
from app.models import Ingredient
from app.schemas import IngredientCreate, IngredientRead

router = APIRouter(prefix="/ingredients", tags=["ingredients"])


@router.post("", response_model=IngredientRead, status_code=status.HTTP_201_CREATED)
async def create_ingredient(payload: IngredientCreate, db: DbSession) -> Ingredient:
    try:
        return await crud.ingredient.create_ingredient(db, payload)
    except DuplicateIngredientError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("", response_model=list[IngredientRead])
async def list_ingredients(
    db: DbSession,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> Sequence[Ingredient]:
    return await crud.ingredient.get_ingredients(db, skip=skip, limit=limit)
