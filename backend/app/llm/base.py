"""Provider-agnostic interfaces for embedding and chat models.

The RAG pipeline depends only on these protocols, so the provider (Ollama today) can be
swapped and tests can plug in deterministic fakes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message of a chat conversation."""

    role: ChatRole
    content: str


class Embedder(Protocol):
    """Turns text into fixed-size vectors for similarity search."""

    @property
    def model_name(self) -> str:
        """Identifier of the embedding model, stored next to every vector."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched (e.g. recipe chunks).

        Args:
            texts: Texts to embed.

        Returns:
            One vector per input text, in input order.

        Raises:
            LLMServiceError: If the embedding service fails.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Args:
            text: The user's query.

        Returns:
            The query vector.

        Raises:
            LLMServiceError: If the embedding service fails.
        """
        ...


class ChatModel(Protocol):
    """Generates a reply to a chat conversation."""

    @property
    def model_name(self) -> str:
        """Identifier of the chat model."""
        ...

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Return the assistant's reply to `messages`.

        Args:
            messages: The conversation so far, usually a system and a user message.

        Returns:
            The reply text.

        Raises:
            LLMServiceError: If the chat service fails.
        """
        ...
