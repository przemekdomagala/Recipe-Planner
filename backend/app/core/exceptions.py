class DuplicateIngredientError(Exception):
    """Raised when creating an ingredient whose name already exists."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Ingredient '{name}' already exists")
        self.name = name


class IngredientsNotFoundError(Exception):
    """Raised when a recipe references ingredient ids that don't exist."""

    def __init__(self, missing_ids: list[int]) -> None:
        super().__init__(f"Unknown ingredient ids: {missing_ids}")
        self.missing_ids = missing_ids
