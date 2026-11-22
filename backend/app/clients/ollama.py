"""Async client for the Ollama REST API (chat + embeddings).

Implements the `Embedder` and `ChatModel` protocols from `app.services.llm`.
See https://docs.ollama.com/api for the endpoints used here.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import BaseModel, SecretStr

from app.core.config import Settings, settings
from app.core.exceptions import (
    LLMAuthError,
    LLMModelNotFoundError,
    LLMQuotaError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.services.llm import ChatMessage


class OllamaStatus(BaseModel):
    """Reachability of the Ollama server and availability of the configured models."""

    base_url: str
    reachable: bool
    chat_model: str
    chat_model_available: bool
    embedding_model: str
    embedding_model_available: bool


def _with_default_tag(model: str) -> str:
    """Canonical model name for comparisons.

    An untagged name means `<name>:latest`, and the `-cloud` suffix (used to reach cloud
    models through a signed-in local Ollama) names the same model as the direct cloud API's
    `gemma4:31b`.
    """
    tagged = model if ":" in model else f"{model}:latest"
    return tagged.removesuffix("-cloud")


class OllamaClient:
    """Thin async wrapper around Ollama's `/api/chat`, `/api/embed` and `/api/tags`.

    Use it as an async context manager (or call `aclose`) so the HTTP connection pool is
    released.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        chat_model: str,
        embedding_model: str,
        embedding_dim: int,
        temperature: float = 0.2,
        think: bool = False,
        max_tokens: int | None = None,
        json_max_tokens: int | None = None,
        timeout: float = 300.0,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Configure the client.

        Args:
            base_url: Ollama server URL, e.g. `https://ollama.com` or `http://localhost:11434`.
            api_key: Ollama Cloud API key, sent as `Authorization: Bearer <key>`; `None` for
                a local server.
            chat_model: Model tag used by `chat`.
            embedding_model: Model tag used by the `embed_*` methods.
            embedding_dim: Expected vector size; mismatching responses are rejected.
            temperature: Sampling temperature for chat.
            think: Enable the model's reasoning mode (ignored by models without one).
            max_tokens: Upper bound on generated tokens (`num_predict`); `None` = no limit.
            json_max_tokens: Like `max_tokens`, for replies constrained to a JSON schema
                (structured output is longer and truncation makes it unparseable).
            timeout: Read timeout in seconds (model loading can be slow).
            query_prefix: Text prepended to search queries before embedding.
            document_prefix: Text prepended to documents before embedding.
            batch_size: Maximum number of texts per `/api/embed` request.
            transport: Optional custom transport (used by tests).
        """
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._temperature = temperature
        self._think = think
        self._max_tokens = max_tokens
        self._json_max_tokens = json_max_tokens
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            transport=transport,
        )

    @classmethod
    def for_chat(cls, config: Settings = settings) -> Self:
        """Build the client used for answer generation (`ollama_*` settings)."""
        return cls._from_settings(config, config.ollama_base_url, config.ollama_api_key)

    @classmethod
    def for_embeddings(cls, config: Settings = settings) -> Self:
        """Build the client used for embeddings (`embedding_*` settings, else `ollama_*`)."""
        return cls._from_settings(config, config.embedding_server_url, config.embedding_server_key)

    @classmethod
    def _from_settings(cls, config: Settings, base_url: str, api_key: SecretStr | None) -> Self:
        """Build a client for `base_url` with every other option taken from `config`."""
        return cls(
            base_url=base_url,
            api_key=api_key.get_secret_value() if api_key else None,
            chat_model=config.llm_model,
            embedding_model=config.embedding_model,
            embedding_dim=config.embedding_dim,
            temperature=config.llm_temperature,
            think=config.llm_think,
            max_tokens=config.llm_max_tokens,
            json_max_tokens=config.llm_json_max_tokens,
            timeout=config.ollama_timeout_seconds,
            query_prefix=config.embedding_query_prefix,
            document_prefix=config.embedding_document_prefix,
            batch_size=config.embedding_batch_size,
        )

    @property
    def base_url(self) -> str:
        """URL of the Ollama server."""
        return str(self._http.base_url)

    @property
    def chat_model(self) -> str:
        """Chat model tag."""
        return self._chat_model

    @property
    def embedding_model(self) -> str:
        """Embedding model tag."""
        return self._embedding_model

    async def __aenter__(self) -> Self:
        """Return the client itself."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def chat(
        self, messages: Sequence[ChatMessage], *, json_schema: dict[str, Any] | None = None
    ) -> str:
        """Return the model's (non-streamed) reply to `messages`.

        Args:
            messages: The conversation so far.
            json_schema: If given, sent as Ollama's `format` so the reply is JSON matching
                the schema; `json_max_tokens` then replaces `max_tokens`.

        Raises:
            LLMUnavailableError: Ollama can't be reached or timed out.
            LLMModelNotFoundError: The chat model isn't installed.
            LLMResponseError: Ollama returned an error or an unexpected payload.
        """
        max_tokens = self._json_max_tokens if json_schema is not None else self._max_tokens
        options: dict[str, float | int] = {"temperature": self._temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self._chat_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": self._think,
            "options": options,
        }
        if json_schema is not None:
            payload["format"] = json_schema
        body = await self._post("/api/chat", payload, model=self._chat_model)
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMResponseError("Ollama chat response has no message content") from exc
        if not isinstance(content, str):
            raise LLMResponseError("Ollama chat response content is not a string")
        return content.strip()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents in batches of `batch_size`, prefixing each with `document_prefix`."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._embed([self._document_prefix + text for text in batch]))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query, prefixed with `query_prefix`."""
        (vector,) = await self._embed([self._query_prefix + text])
        return vector

    async def status(self) -> OllamaStatus:
        """Check whether Ollama is reachable and the configured models are installed."""
        installed: set[str] = set()
        reachable = True
        try:
            response = await self._http.get("/api/tags", timeout=5.0)
            response.raise_for_status()
            installed = {_with_default_tag(m["name"]) for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            reachable = False
        return OllamaStatus(
            base_url=self.base_url,
            reachable=reachable,
            chat_model=self._chat_model,
            chat_model_available=_with_default_tag(self._chat_model) in installed,
            embedding_model=self._embedding_model,
            embedding_model_available=_with_default_tag(self._embedding_model) in installed,
        )

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        """Call `/api/embed` for one batch and validate the vectors' count and size."""
        if not inputs:
            return []
        body = await self._post(
            "/api/embed",
            {"model": self._embedding_model, "input": inputs, "truncate": True},
            model=self._embedding_model,
        )
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise LLMResponseError("Ollama returned an unexpected number of embeddings")
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != self._embedding_dim:
                size = len(vector) if isinstance(vector, list) else "?"
                raise LLMResponseError(
                    f"Embedding model '{self._embedding_model}' returned {size}-dim vectors, "
                    f"but EMBEDDING_DIM is {self._embedding_dim}"
                )
        return [[float(x) for x in vector] for vector in embeddings]

    async def _post(self, path: str, payload: dict[str, Any], *, model: str) -> dict[str, Any]:
        """POST JSON to Ollama and map transport/HTTP failures to domain exceptions."""
        try:
            response = await self._http.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(f"Ollama at {self.base_url} timed out") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(f"Ollama is unreachable at {self.base_url}") from exc

        if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise LLMAuthError(
                f"Ollama at {self.base_url} rejected the API key (set OLLAMA_API_KEY)"
            )
        if response.status_code in (httpx.codes.PAYMENT_REQUIRED, httpx.codes.TOO_MANY_REQUESTS):
            raise LLMQuotaError(
                f"Ollama at {self.base_url} refused the request ({response.status_code}): "
                f"{_error_message(response)}"
            )

        if response.is_success:
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMResponseError("Ollama returned invalid JSON") from exc
            if not isinstance(body, dict):
                raise LLMResponseError("Ollama returned an unexpected payload")
            return body

        error = _error_message(response)
        if response.status_code == httpx.codes.NOT_FOUND and "not found" in error.lower():
            raise LLMModelNotFoundError(model)
        raise LLMResponseError(f"Ollama error {response.status_code}: {error}")


def _error_message(response: httpx.Response) -> str:
    """Extract Ollama's `{"error": ...}` message, falling back to the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, str) else response.text
