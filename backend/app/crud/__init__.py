"""Database access functions, grouped by resource."""

from app.crud import chunk, fridge, ingredient, profile, recipe

__all__ = ["chunk", "fridge", "ingredient", "profile", "recipe"]
