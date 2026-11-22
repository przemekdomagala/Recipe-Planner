"""Provider-agnostic interfaces for embedding and chat models.

Services depend on these protocols rather than on a concrete client, so the backend
(currently Ollama) can be swapped and tests can plug in deterministic fakes.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message of a chat conversation."""

    role: ChatRole
    content: str


class Embedder(Protocol):
    """Turns text into dense vectors for similarity search."""

    @property
    def embedding_model(self) -> str:
        """Identifier of the embedding model, stored next to every vector it produced."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts that will be stored and searched (one vector per text, same order)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class ChatModel(Protocol):
    """Generates a reply for a chat conversation."""

    @property
    def chat_model(self) -> str:
        """Identifier of the chat model."""
        ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's reply to `messages`.

        Args:
            messages: The conversation so far.
            json_schema: Constrain the reply to JSON matching this schema (structured output).
                The caller still validates the reply: small models may not comply.
            max_tokens: Override the model's default cap on generated tokens.
        """
        ...
