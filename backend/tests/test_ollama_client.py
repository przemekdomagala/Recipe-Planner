"""Tests for the Ollama client, against a mocked HTTP transport."""

import json
from collections.abc import Callable

import httpx
import pytest
from app.clients.ollama import OllamaClient
from app.core.config import Settings
from app.core.exceptions import (
    LLMAuthError,
    LLMModelNotFoundError,
    LLMQuotaError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.services.llm import ChatMessage

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, *, dim: int = 3, batch_size: int = 32) -> OllamaClient:
    """Client for a fake Ollama Cloud with an API key and nomic-style prefixes."""
    return OllamaClient(
        base_url="https://ollama.test/",
        api_key="secret",
        chat_model="gemma4:31b",
        embedding_model="nomic-embed-text",
        embedding_dim=dim,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        batch_size=batch_size,
        max_tokens=200,
        transport=httpx.MockTransport(handler),
    )


async def test_chat_sends_auth_and_returns_content() -> None:
    """Chat posts a non-streaming request with the bearer key and returns the reply."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer like Ollama."""
        seen.append(request)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hi! "}})

    async with make_client(handler) as client:
        reply = await client.chat([ChatMessage("user", "Hello")])

    assert reply == "Hi!"
    request = seen[0]
    assert request.url.path == "/api/chat"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "gemma4:31b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"]["num_predict"] == 200
    assert body["messages"] == [{"role": "user", "content": "Hello"}]


async def test_no_auth_header_without_key() -> None:
    """A local server (no key) gets no Authorization header."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request."""
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = OllamaClient(
        base_url="http://localhost:11434",
        chat_model="m",
        embedding_model="e",
        embedding_dim=3,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        await client.chat([ChatMessage("user", "x")])

    assert "Authorization" not in seen[0].headers


async def test_chat_with_json_schema_sets_format() -> None:
    """A JSON schema is sent as `format`, with the structured-output token cap."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with JSON."""
        seen.append(request)
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    client = OllamaClient(
        base_url="http://localhost:11434",
        chat_model="m",
        embedding_model="e",
        embedding_dim=3,
        max_tokens=200,
        json_max_tokens=900,
        transport=httpx.MockTransport(handler),
    )
    async with client:
        reply = await client.chat([ChatMessage("user", "x")], json_schema=schema)

    assert reply == '{"ok": true}'
    body = json.loads(seen[0].content)
    assert body["format"] == schema
    assert body["options"]["num_predict"] == 900


async def test_embed_documents_batches_and_prefixes() -> None:
    """Documents are prefixed and sent in batches; vectors come back in order."""
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return one vector per input, encoding the global index."""
        inputs = json.loads(request.content)["input"]
        offset = sum(len(b) for b in batches)
        batches.append(inputs)
        return httpx.Response(
            200, json={"embeddings": [[float(offset + i), 0.0, 0.0] for i in range(len(inputs))]}
        )

    async with make_client(handler, batch_size=2) as client:
        vectors = await client.embed_documents(["a", "b", "c"])
        query = await client.embed_query("q")

    assert batches == [
        ["search_document: a", "search_document: b"],
        ["search_document: c"],
        ["search_query: q"],
    ]
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]
    assert query == [3.0, 0.0, 0.0]


async def test_embedding_dimension_mismatch_is_rejected() -> None:
    """Vectors that don't match EMBEDDING_DIM raise instead of corrupting the index."""
    handler: Handler = lambda request: httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})  # noqa: E731

    async with make_client(handler, dim=3) as client:
        with pytest.raises(LLMResponseError, match="2-dim"):
            await client.embed_query("q")


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(401, json={"error": "unauthorized"}), LLMAuthError),
        (httpx.Response(402, json={"error": "Add usage credits"}), LLMQuotaError),
        (httpx.Response(429, json={"error": "rate limited"}), LLMQuotaError),
        (
            httpx.Response(404, json={"error": "model 'gemma4:31b' not found"}),
            LLMModelNotFoundError,
        ),
        (httpx.Response(500, json={"error": "boom"}), LLMResponseError),
        (httpx.Response(200, json={"unexpected": True}), LLMResponseError),
    ],
)
async def test_http_errors_map_to_domain_errors(
    response: httpx.Response, error: type[Exception]
) -> None:
    """Ollama's error responses become specific `LLMError` subclasses."""
    async with make_client(lambda request: response) as client:
        with pytest.raises(error):
            await client.chat([ChatMessage("user", "x")])


async def test_connection_failure_is_unavailable() -> None:
    """Network errors become `LLMUnavailableError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate a refused connection."""
        raise httpx.ConnectError("refused", request=request)

    async with make_client(handler) as client:
        with pytest.raises(LLMUnavailableError):
            await client.chat([ChatMessage("user", "x")])


def test_embedding_settings_fall_back_to_chat_server() -> None:
    """Embeddings reuse the chat server and key unless configured separately."""
    shared = Settings(
        _env_file=None,
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        embedding_base_url=None,
        embedding_api_key=None,
    )
    separate = Settings(
        _env_file=None,
        ollama_base_url="https://ollama.com",
        ollama_api_key="k",
        embedding_base_url="http://localhost:11434",
        embedding_api_key=None,
    )

    assert shared.embedding_server_url == "https://ollama.com"
    assert shared.embedding_server_key is not None
    assert shared.embedding_server_key.get_secret_value() == "k"
    assert separate.embedding_server_url == "http://localhost:11434"
    assert separate.embedding_server_key is None  # the cloud key never leaks elsewhere


def test_blank_settings_count_as_unset() -> None:
    """Empty env values (as Compose passes them) don't disable the fallbacks."""
    config = Settings(
        _env_file=None, ollama_api_key="k", embedding_base_url="", embedding_api_key=" "
    )

    assert config.embedding_server_url == config.ollama_base_url
    assert config.embedding_server_key is not None
    assert config.embedding_server_key.get_secret_value() == "k"
