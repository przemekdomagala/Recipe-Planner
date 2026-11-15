"""Ollama implementations of the `Embedder` and `ChatModel` protocols (REST API over httpx)."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Self

import httpx

from app.core.config import Settings
from app.core.exceptions import LLMServiceError
from app.llm.base import ChatMessage


async def _post_json(http: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to Ollama and return the decoded reply, translating failures.

    Raises:
        LLMServiceError: On connection errors, timeouts and non-2xx replies; the message
            includes Ollama's own error text (e.g. "model ... not found").
    """
    try:
        response = await http.post(path, json=payload)
    except httpx.TimeoutException as exc:
        raise LLMServiceError(f"Ollama request to {path} timed out") from exc
    except httpx.HTTPError as exc:
        raise LLMServiceError(f"Cannot reach Ollama at {http.base_url}: {exc}") from exc
    if response.is_error:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        hint = " (pull it with `ollama pull <model>`)" if response.status_code == 404 else ""
        raise LLMServiceError(f"Ollama error {response.status_code}: {detail}{hint}")
    result: dict[str, Any] = response.json()
    return result


def _with_tag(model: str) -> str:
    """Return `model` with Ollama's implicit ":latest" tag added if it has no tag."""
    return model if ":" in model else f"{model}:latest"


class OllamaEmbedder:
    """`Embedder` backed by Ollama's `/api/embed` endpoint."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        dimensions: int,
        document_prefix: str = "",
        query_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        """Configure the embedder.

        Args:
            http: HTTP client whose `base_url` points at the Ollama server.
            model: Embedding model, e.g. "nomic-embed-text".
            dimensions: Expected vector size; replies of a different size are rejected
                because they would not fit the vector column.
            document_prefix: Prepended to every document before embedding.
            query_prefix: Prepended to every query before embedding.
            batch_size: Maximum number of texts sent in one request.
        """
        self._http = http
        self._model = model
        self._dimensions = dimensions
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        """Name of the Ollama embedding model."""
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (with the document prefix) in batches of at most `batch_size`.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per input text, in input order.

        Raises:
            LLMServiceError: If Ollama fails or returns vectors of the wrong shape.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed([self._document_prefix + text for text in batch]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (with the query prefix).

        Args:
            text: The user's query.

        Raises:
            LLMServiceError: If Ollama fails or returns a vector of the wrong shape.
        """
        return (await self._embed([self._query_prefix + text]))[0]

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        """Call `/api/embed` for one batch and validate the vectors' count and size."""
        body = await _post_json(self._http, "/api/embed", {"model": self._model, "input": inputs})
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise LLMServiceError("Ollama returned an unexpected number of embeddings")
        for vector in embeddings:
            if len(vector) != self._dimensions:
                raise LLMServiceError(
                    f"Embedding model '{self._model}' returned {len(vector)}-dimensional "
                    f"vectors but EMBEDDING_DIM is {self._dimensions}"
                )
        return embeddings


class OllamaChatModel:
    """`ChatModel` backed by Ollama's `/api/chat` endpoint (non-streaming)."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        model: str,
        temperature: float = 0.2,
        num_ctx: int | None = None,
    ) -> None:
        """Configure the chat model.

        Args:
            http: HTTP client whose `base_url` points at the Ollama server.
            model: Chat model, e.g. "llama3.2:3b".
            temperature: Sampling temperature.
            num_ctx: Context window size to request, or `None` for the model default.
        """
        self._http = http
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx

    @property
    def model_name(self) -> str:
        """Name of the Ollama chat model."""
        return self._model

    async def chat(self, messages: Sequence[ChatMessage]) -> str:
        """Return the model's reply to `messages`.

        Args:
            messages: The conversation to answer.

        Raises:
            LLMServiceError: If Ollama fails or the reply has no text.
        """
        options: dict[str, Any] = {"temperature": self._temperature}
        if self._num_ctx is not None:
            options["num_ctx"] = self._num_ctx
        body = await _post_json(
            self._http,
            "/api/chat",
            {
                "model": self._model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": options,
            },
        )
        content = body.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMServiceError(f"Ollama model '{self._model}' returned an empty reply")
        return content.strip()


class Ollama:
    """Facade bundling the configured embedder and chat model of one Ollama server.

    The caller owns the `httpx.AsyncClient` (base URL, timeout, lifecycle), which keeps
    connection pooling under the application's control and lets tests inject a mock
    transport.
    """

    def __init__(
        self, http: httpx.AsyncClient, *, embedder: OllamaEmbedder, chat_model: OllamaChatModel
    ) -> None:
        """Bundle the models served by the Ollama server behind `http`.

        Args:
            http: HTTP client whose `base_url` points at the Ollama server.
            embedder: Embedding model using the same server.
            chat_model: Chat model using the same server.
        """
        self._http = http
        self.embedder = embedder
        self.chat_model = chat_model

    @classmethod
    def from_settings(cls, http: httpx.AsyncClient, settings: Settings) -> Self:
        """Build the embedder and chat model configured in the application settings.

        Args:
            http: HTTP client whose `base_url` points at the Ollama server.
            settings: Application settings.
        """
        return cls(
            http,
            embedder=OllamaEmbedder(
                http,
                model=settings.ollama_embedding_model,
                dimensions=settings.embedding_dim,
                document_prefix=settings.embedding_document_prefix,
                query_prefix=settings.embedding_query_prefix,
                batch_size=settings.embedding_batch_size,
            ),
            chat_model=OllamaChatModel(
                http,
                model=settings.ollama_chat_model,
                temperature=settings.llm_temperature,
                num_ctx=settings.ollama_num_ctx,
            ),
        )

    async def list_models(self) -> list[str]:
        """Return the names of the models pulled on the Ollama server.

        Raises:
            LLMServiceError: If Ollama is unreachable.
        """
        try:
            response = await self._http.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMServiceError(f"Cannot reach Ollama at {self._http.base_url}: {exc}") from exc
        return [model["name"] for model in response.json().get("models", [])]

    async def missing_models(self) -> list[str]:
        """Return the configured models that are not pulled on the server.

        Ollama reports untagged names with the implicit ":latest" tag, so both spellings
        are treated as equal.

        Raises:
            LLMServiceError: If Ollama is unreachable.
        """
        available = {_with_tag(name) for name in await self.list_models()}
        required = dict.fromkeys((self.embedder.model_name, self.chat_model.model_name))
        return [name for name in required if _with_tag(name) not in available]


@asynccontextmanager
async def ollama_client(settings: Settings) -> AsyncIterator[Ollama]:
    """Yield an `Ollama` facade backed by an HTTP client that is closed on exit.

    Args:
        settings: Application settings (base URL, timeout, models).
    """
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds
    ) as http:
        yield Ollama.from_settings(http, settings)
