"""Diet profile endpoints (`/api/profile`)."""

from fastapi import APIRouter

from app import crud
from app.api.deps import DbSession
from app.schemas import DietOption, DietProfileData, LabeledOption, ProfileOptions
from app.services.dietary import ALLERGEN_LABELS, DIET_FORBIDS, DIET_LABELS

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=DietProfileData)
async def get_profile(db: DbSession) -> DietProfileData:
    """Return the diet profile (empty until it is first saved)."""
    return DietProfileData.model_validate(await crud.profile.get_profile(db))


@router.put("", response_model=DietProfileData)
async def update_profile(payload: DietProfileData, db: DbSession) -> DietProfileData:
    """Replace the diet profile."""
    return DietProfileData.model_validate(await crud.profile.update_profile(db, payload))


@router.get("/options", response_model=ProfileOptions)
async def profile_options() -> ProfileOptions:
    """List the supported diets (with what they rule out) and allergens."""
    return ProfileOptions(
        diets=[
            DietOption(value=diet, label=label, forbids=sorted(DIET_FORBIDS[diet]))
            for diet, label in DIET_LABELS.items()
        ],
        allergens=[
            LabeledOption(value=allergen, label=label)
            for allergen, label in ALLERGEN_LABELS.items()
        ],
    )
