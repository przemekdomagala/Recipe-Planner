"""Domain exceptions raised below the API layer and mapped to HTTP errors by routers."""


class DuplicateIngredientError(Exception):
    """Raised when creating an ingredient whose name already exists."""

    def __init__(self, name: str) -> None:
        """Store the conflicting ingredient name.

        Args:
            name: The (normalized) name that already exists.
        """
        super().__init__(f"Ingredient '{name}' already exists")
        self.name = name


class IngredientsNotFoundError(Exception):
    """Raised when a recipe references ingredient ids that don't exist."""

    def __init__(self, missing_ids: list[int]) -> None:
        """Store the ingredient ids that could not be found.

        Args:
            missing_ids: Sorted list of unknown ingredient ids.
        """
        super().__init__(f"Unknown ingredient ids: {missing_ids}")
        self.missing_ids = missing_ids


class LLMServiceError(Exception):
    """Raised when the language model service is unreachable or returns an unusable reply."""


class RecipeSourceError(Exception):
    """Raised when an external recipe source (e.g. TheMealDB) cannot be read."""
