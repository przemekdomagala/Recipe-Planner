"""Database access for the (single) diet profile."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DietProfile
from app.models.diet_profile import DEFAULT_PROFILE_ID
from app.schemas import DietProfileData


async def get_profile(db: AsyncSession) -> DietProfile:
    """Return the diet profile, creating an empty one on first use.

    Args:
        db: Database session.

    Returns:
        The persisted profile.
    """
    profile = await db.get(DietProfile, DEFAULT_PROFILE_ID)
    if profile is None:
        profile = DietProfile(
            id=DEFAULT_PROFILE_ID, diets=[], allergens=[], excluded_ingredients=[]
        )
        db.add(profile)
        await db.commit()
    return profile


async def update_profile(db: AsyncSession, data: DietProfileData) -> DietProfile:
    """Replace the diet profile's constraints with `data`.

    Args:
        db: Database session.
        data: The complete new set of constraints.

    Returns:
        The updated profile.
    """
    profile = await get_profile(db)
    profile.diets = [diet.value for diet in data.diets]
    profile.allergens = [allergen.value for allergen in data.allergens]
    profile.excluded_ingredients = list(data.excluded_ingredients)
    await db.commit()
    return profile
