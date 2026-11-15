"""Retrieval-augmented generation over the recipe knowledge base.

Pipeline: `chunking` splits recipes into texts, `indexing` embeds and stores them in
`recipe_chunks` (pgvector), `retrieval` finds the recipes nearest to a query, and
`generation` grounds the chat model's answer on them using the `prompts`.
"""

from app.rag.chunking import ChunkKind, RecipeDocument, TextChunk, chunk_recipe, split_text
from app.rag.generation import Answer, answer_question, generate_answer
from app.rag.indexing import (
    IndexStats,
    ReindexResult,
    get_index_stats,
    index_recipe,
    reindex_recipes,
)
from app.rag.retrieval import RecipeHit, RetrievedChunk, retrieve_recipes

__all__ = [
    "Answer",
    "ChunkKind",
    "IndexStats",
    "RecipeDocument",
    "RecipeHit",
    "ReindexResult",
    "RetrievedChunk",
    "TextChunk",
    "answer_question",
    "chunk_recipe",
    "generate_answer",
    "get_index_stats",
    "index_recipe",
    "reindex_recipes",
    "retrieve_recipes",
    "split_text",
]
