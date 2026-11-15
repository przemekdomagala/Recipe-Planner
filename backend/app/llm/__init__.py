"""Language model integration: provider-agnostic protocols and the Ollama implementation."""

from app.llm.base import ChatMessage, ChatModel, ChatRole, Embedder
from app.llm.ollama import Ollama, OllamaChatModel, OllamaEmbedder, ollama_client

__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatRole",
    "Embedder",
    "Ollama",
    "OllamaChatModel",
    "OllamaEmbedder",
    "ollama_client",
]
