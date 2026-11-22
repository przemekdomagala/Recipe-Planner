"""Deterministic stand-ins for the embedding and chat models."""

import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from app.core.config import settings
from app.services.llm import ChatMessage

_WORD = re.compile(r"[a-z]+")


class FakeEmbedder:
    """Bag-of-words hashing embedder: texts sharing words get similar vectors."""

    def __init__(self, model: str = "fake-embed", dim: int = settings.embedding_dim) -> None:
        """Create an embedder producing `dim`-sized vectors tagged with `model`."""
        self._model = model
        self._dim = dim
        self.calls = 0

    @property
    def embedding_model(self) -> str:
        """Model name stored on the chunks."""
        return self._model

    def vector(self, text: str) -> list[float]:
        """Embed one text: each word adds 1 to a hashed bucket, then L2-normalize."""
        values = [0.0] * self._dim
        for word in _WORD.findall(text.lower()):
            bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            values[bucket % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (counts one call per batch)."""
        self.calls += 1
        return [self.vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query."""
        return self.vector(text)


class FakeChatModel:
    """Chat model that records the conversation and returns a canned reply.

    `reply` may also be a function of the messages, for replies that depend on the prompt.
    Set `error` to make every call raise it instead (e.g. a missing model).
    """

    def __init__(
        self,
        reply: str | Callable[[Sequence[ChatMessage]], str] = (
            "Try the Teriyaki Chicken Casserole [1]."
        ),
    ) -> None:
        """Create a model that answers `reply`."""
        self.reply = reply
        self.error: Exception | None = None
        self.received: list[list[ChatMessage]] = []
        self.schemas: list[Mapping[str, Any] | None] = []

    @property
    def chat_model(self) -> str:
        """Model name reported in responses."""
        return "fake-chat"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Record `messages` and the requested schema, then return the canned reply."""
        self.received.append(list(messages))
        self.schemas.append(json_schema)
        if self.error is not None:
            raise self.error
        return self.reply if isinstance(self.reply, str) else self.reply(messages)
