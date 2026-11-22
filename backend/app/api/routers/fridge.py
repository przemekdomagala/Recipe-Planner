"""Fridge endpoints (`/api/fridge`): items at hand and recipe suggestions for them."""

from fastapi import APIRouter, HTTPException, Response, status

from app import crud
from app.api.deps import ChatModelDep, DbSession, EmbedderDep
from app.core.exceptions import DuplicateFridgeItemError
from app.models import FridgeItem
from app.schemas import (
    FridgeItemCreate,
    FridgeItemRead,
    FridgeItemUpdate,
    SuggestionRequest,
    SuggestionResponse,
)
from app.services.fridge import suggest_recipes

router = APIRouter(prefix="/fridge", tags=["fridge"])


async def _get_item_or_404(db: DbSession, item_id: int) -> FridgeItem:
    """Return the fridge item or raise a 404."""
    item = await crud.fridge.get_fridge_item(db, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fridge item not found")
    return item


@router.get("/items", response_model=list[FridgeItemRead])
async def list_items(db: DbSession) -> list[FridgeItem]:
    """List the fridge contents, soonest expiry first."""
    return await crud.fridge.get_fridge_items(db)


@router.post("/items", response_model=FridgeItemRead, status_code=status.HTTP_201_CREATED)
async def add_item(payload: FridgeItemCreate, db: DbSession) -> FridgeItem:
    """Add an ingredient to the fridge, creating the ingredient if it's new.

    Raises:
        HTTPException: 409 if the ingredient is already in the fridge.
    """
    try:
        return await crud.fridge.add_fridge_item(db, payload)
    except DuplicateFridgeItemError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.patch("/items/{item_id}", response_model=FridgeItemRead)
async def update_item(item_id: int, payload: FridgeItemUpdate, db: DbSession) -> FridgeItem:
    """Change the amount or expiry date of a fridge item.

    Raises:
        HTTPException: 404 if the item doesn't exist.
    """
    item = await _get_item_or_404(db, item_id)
    return await crud.fridge.update_fridge_item(db, item, payload)


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int, db: DbSession) -> Response:
    """Remove an item from the fridge.

    Raises:
        HTTPException: 404 if the item doesn't exist.
    """
    await crud.fridge.delete_fridge_item(db, await _get_item_or_404(db, item_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/suggestions", response_model=SuggestionResponse)
async def suggestions(
    payload: SuggestionRequest, db: DbSession, embedder: EmbedderDep, chat_model: ChatModelDep
) -> SuggestionResponse:
    """Suggest recipes for the fridge contents, adapted to the diet profile.

    Never fails because of the LLM: without it the rule-based ranking is returned and the
    fallback is explained in `notes`.
    """
    return await suggest_recipes(db, payload, embedder, chat_model)
