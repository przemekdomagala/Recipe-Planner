"""Recipe-ingredient association ORM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.ingredient import Ingredient
    from app.models.recipe import Recipe


class RecipeIngredient(Base):
    """Association object linking a recipe to an ingredient, with an optional amount.

    `quantity` and `unit` are the machine-readable amount (`unit` is canonical, e.g.
    "tbsp"); `measure` keeps the original free-text amount of imported recipes
    (e.g. "1 clove peeled crushed"), which often carries more than a number and a unit.
    """

    __tablename__ = "recipe_ingredients"

    recipe_id: Mapped[int] = mapped_column(
        ForeignKey("recipes.id", ondelete="CASCADE"), primary_key=True
    )
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="RESTRICT"), primary_key=True
    )
    # Order in which the ingredient is listed in the recipe.
    position: Mapped[int] = mapped_column(default=0)
    quantity: Mapped[float | None] = mapped_column(default=None)
    unit: Mapped[str | None] = mapped_column(String(32), default=None)
    measure: Mapped[str | None] = mapped_column(String(200), default=None)

    recipe: Mapped[Recipe] = relationship(back_populates="ingredients")
    ingredient: Mapped[Ingredient] = relationship()
