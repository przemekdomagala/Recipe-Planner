"""Deterministic stand-ins for the embedding and chat models."""

import hashlib
import math
import re
from collections.abc import Sequence

from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.llm import ChatMessage

_WORD = re.compile(r"[a-z]+")


class FakeEmbedder:
    """Hashed bag-of-words embedder: texts sharing words get similar vectors.

    Good enough to make nearest-neighbour ranking meaningful in tests without a model.
    """

    model_name = "fake-embedder"

    def __init__(self, dimensions: int = settings.embedding_dim) -> None:
        """Create the embedder.

        Args:
            dimensions: Vector size; must match the vector column.
        """
        self.dimensions = dimensions
        self.calls = 0
        self.fail = False

    def _vector(self, text: str) -> list[float]:
        """Return the L2-normalized hashed word counts of `text`."""
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents, or raise if `fail` is set."""
        self.calls += 1
        if self.fail:
            raise LLMServiceError("embedder down")
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query, or raise if `fail` is set."""
        self.calls += 1
        if self.fail:
            raise LLMServiceError("embedder down")
        return self._vector(text)


class FakeChatModel:
    """Chat model that records the conversations it receives and returns a fixed reply."""

    model_name = "fake-chat"

    def __init__(self, reply: str = "Try the Chicken Curry.") -> None:
        """Create the model.

        Args:
            reply: Text returned by every `chat` call.
        """
        self.reply = reply
        self.received: list[list[ChatMessage]] = []
        self.fail = False

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Record `messages` and return the fixed reply, or raise if `fail` is set."""
        if self.fail:
            raise LLMServiceError("chat model down")
        self.received.append(list(messages))
        return self.reply
