"""'What's in the Fridge' endpoints (`/api/fridge`): constraint-checked RAG suggestions."""

from fastapi import APIRouter

from app.api.deps import ChatModelDep, DbSession, EmbedderDep
from app.schemas import FridgeRequest, FridgeResponse
from app.services.fridge import suggest_recipes

router = APIRouter(prefix="/fridge", tags=["fridge"])


@router.post("/suggestions", response_model=FridgeResponse)
async def suggestions(
    payload: FridgeRequest, db: DbSession, embedder: EmbedderDep, chat_model: ChatModelDep
) -> FridgeResponse:
    """Rank recipes for the fridge contents and adapt them to the user's restrictions.

    Every suggestion is checked against the allergens, diet and avoided ingredients after
    generation; recipes that still conflict are listed under `excluded` with the reasons.
    Returns 503 if the LLM backend is unreachable, unauthorized or lacks the model.
    """
    return await suggest_recipes(db, payload, embedder, chat_model)
