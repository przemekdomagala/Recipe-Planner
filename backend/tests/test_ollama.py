"""Tests for the Ollama client, against a mocked HTTP transport."""

import json
from collections.abc import Callable

import httpx
import pytest
from app.core.exceptions import LLMServiceError
from app.llm import ChatMessage, Ollama, OllamaChatModel, OllamaEmbedder

Handler = Callable[[httpx.Request], httpx.Response]


def _http(handler: Handler) -> httpx.AsyncClient:
    """Build an HTTP client whose requests are answered by `handler`."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ollama.test")


async def test_embed_documents_batches_and_prefixes() -> None:
    """Documents are sent in batches with the document prefix; order is preserved."""
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a vector whose first value is the input's length."""
        body = json.loads(request.content)
        assert request.url.path == "/api/embed"
        assert body["model"] == "embed-model"
        batches.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[len(t), 0.0] for t in body["input"]]})

    embedder = OllamaEmbedder(
        _http(handler), model="embed-model", dimensions=2, document_prefix="doc: ", batch_size=2
    )

    vectors = await embedder.embed_documents(["a", "bb", "ccc"])

    assert batches == [["doc: a", "doc: bb"], ["doc: ccc"]]
    assert vectors == [[6.0, 0.0], [7.0, 0.0], [8.0, 0.0]]


async def test_embed_query_uses_query_prefix() -> None:
    """Queries get the query prefix."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the input and return a fixed vector."""
        seen.extend(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0]]})

    embedder = OllamaEmbedder(_http(handler), model="m", dimensions=2, query_prefix="q: ")

    assert await embedder.embed_query("soup") == [1.0, 2.0]
    assert seen == ["q: soup"]


async def test_embedding_dimension_mismatch_is_rejected() -> None:
    """Vectors that would not fit the vector column raise a descriptive error."""
    embedder = OllamaEmbedder(
        _http(lambda request: httpx.Response(200, json={"embeddings": [[1.0, 2.0, 3.0]]})),
        model="m",
        dimensions=2,
    )

    with pytest.raises(LLMServiceError, match="3-dimensional"):
        await embedder.embed_query("x")


async def test_chat_sends_messages_and_options() -> None:
    """The conversation, model and options are sent; the reply text is returned."""
    sent: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request body and reply."""
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": " Hi! "}})

    chat = OllamaChatModel(_http(handler), model="chat-model", temperature=0.1, num_ctx=4096)

    reply = await chat.chat([ChatMessage("system", "Be brief."), ChatMessage("user", "Hello")])

    assert reply == "Hi!"
    assert sent == {
        "model": "chat-model",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }


async def test_chat_rejects_empty_reply() -> None:
    """A blank reply is treated as a failure."""
    chat = OllamaChatModel(
        _http(lambda request: httpx.Response(200, json={"message": {"content": "  "}})),
        model="m",
    )

    with pytest.raises(LLMServiceError, match="empty reply"):
        await chat.chat([ChatMessage("user", "Hello")])


async def test_missing_model_error_mentions_pull() -> None:
    """Ollama's 404 for an unknown model is surfaced with a hint."""
    chat = OllamaChatModel(
        _http(lambda request: httpx.Response(404, json={"error": "model 'x' not found"})),
        model="x",
    )

    with pytest.raises(LLMServiceError, match=r"model 'x' not found.*ollama pull"):
        await chat.chat([ChatMessage("user", "Hello")])


async def test_connection_error_is_wrapped() -> None:
    """Transport failures become `LLMServiceError`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Simulate a server that is down."""
        raise httpx.ConnectError("connection refused", request=request)

    embedder = OllamaEmbedder(_http(handler), model="m", dimensions=2)

    with pytest.raises(LLMServiceError, match="Cannot reach Ollama"):
        await embedder.embed_query("x")


async def test_missing_models_treats_untagged_names_as_latest() -> None:
    """Untagged model names match their ":latest" tag; absent models are reported."""
    http = _http(
        lambda request: httpx.Response(
            200, json={"models": [{"name": "nomic-embed-text:latest"}, {"name": "other:1b"}]}
        )
    )
    ollama = Ollama(
        http,
        embedder=OllamaEmbedder(http, model="nomic-embed-text", dimensions=768),
        chat_model=OllamaChatModel(http, model="llama3.2:3b"),
    )

    assert await ollama.missing_models() == ["llama3.2:3b"]
