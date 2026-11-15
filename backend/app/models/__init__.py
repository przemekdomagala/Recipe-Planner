"""SQLAlchemy ORM models.

Every model must be imported here so it is registered on `Base.metadata` (and thus
picked up by `create_all`).
"""

from app.models.ingredient import Ingredient
from app.models.recipe import USER_SOURCE, Recipe
from app.models.recipe_chunk import RecipeChunk
from app.models.recipe_ingredient import RecipeIngredient

__all__ = ["USER_SOURCE", "Ingredient", "Recipe", "RecipeChunk", "RecipeIngredient"]
