"""Domain exceptions raised by the crud layer and mapped to HTTP errors by routers."""


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


class LLMError(Exception):
    """Base class for failures talking to the LLM / embedding backend (Ollama)."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM backend can't be reached or doesn't answer in time."""


class LLMModelNotFoundError(LLMError):
    """Raised when the configured model isn't installed in the LLM backend."""

    def __init__(self, model: str) -> None:
        """Store the missing model name.

        Args:
            model: Model tag that the backend doesn't know, e.g. `gemma4:31b`.
        """
        super().__init__(f"Model '{model}' is not available; pull it with `ollama pull {model}`")
        self.model = model


class LLMResponseError(LLMError):
    """Raised when the LLM backend answers with an error or an unexpected payload."""


class RecipeSourceError(Exception):
    """Raised when an external recipe source (e.g. TheMealDB) can't be read."""


class LLMAuthError(LLMError):
    """Raised when the LLM backend rejects the API key (missing or invalid)."""


class LLMQuotaError(LLMError):
    """Raised when the LLM backend refuses for billing/quota reasons (HTTP 402/429)."""
