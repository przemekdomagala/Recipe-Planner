"""Diet profile ORM model: the user's dietary constraints."""

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# There are no user accounts yet, so the app keeps a single profile row.
DEFAULT_PROFILE_ID = 1


class DietProfile(Base):
    """Diets, allergens and excluded ingredients that recipe suggestions must respect.

    Values are stored as the string values of `app.services.dietary.Diet` / `Allergen`.
    """

    __tablename__ = "diet_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    diets: Mapped[list[str]] = mapped_column(JSON, default=list)
    allergens: Mapped[list[str]] = mapped_column(JSON, default=list)
    excluded_ingredients: Mapped[list[str]] = mapped_column(JSON, default=list)
