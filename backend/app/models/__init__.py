"""SQLAlchemy ORM models.

Every model must be imported here so it is registered on `Base.metadata` (and thus
picked up by `create_all`).
"""

from app.models.diet_profile import DietProfile
from app.models.fridge_item import FridgeItem
from app.models.ingredient import Ingredient
from app.models.recipe import Recipe
from app.models.recipe_chunk import RecipeChunk
from app.models.recipe_ingredient import RecipeIngredient

__all__ = ["DietProfile", "FridgeItem", "Ingredient", "Recipe", "RecipeChunk", "RecipeIngredient"]
