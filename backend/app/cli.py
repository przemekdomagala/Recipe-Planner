"""Command-line tasks for the knowledge base.

Usage (from `backend/`, or `docker compose exec backend python -m app.cli ...`)::

    uv run python -m app.cli ingest-mealdb              # import + index all of TheMealDB
    uv run python -m app.cli ingest-mealdb --letters ab # only meals starting with a/b
    uv run python -m app.cli reindex [--force]          # (re)embed stale or all recipes
    uv run python -m app.cli llm-status                 # check the Ollama connection
"""

import argparse
import asyncio
import logging
import sys

from app.clients.ollama import OllamaClient
from app.clients.themealdb import ALL_LETTERS, TheMealDBClient
from app.core.exceptions import LLMError, RecipeSourceError
from app.db.session import AsyncSessionLocal, init_db
from app.services.indexing import reindex_recipes
from app.services.ingestion import ingest_recipes


def _progress(done: int, total: int) -> None:
    """Print a single-line progress counter."""
    end = "\n" if done == total else ""
    print(f"\r  processed {done}/{total}", end=end, flush=True)


async def _ingest_mealdb(letters: str, limit: int | None, force: bool) -> None:
    """Fetch TheMealDB recipes and ingest them."""
    await init_db()
    async with TheMealDBClient() as source:
        recipes = await source.fetch_all(letters)
    if limit is not None:
        recipes = recipes[:limit]
    print(f"Fetched {len(recipes)} recipes from TheMealDB")
    async with OllamaClient.for_embeddings() as embedder, AsyncSessionLocal() as db:
        report = await ingest_recipes(
            db, recipes, embedder, force_reindex=force, on_progress=_progress
        )
    print(f"Created {report.created}, updated {report.updated}, (re)indexed {report.indexed}")


async def _reindex(force: bool) -> None:
    """Re-embed recipes whose chunks are missing or stale."""
    await init_db()
    async with OllamaClient.for_embeddings() as embedder, AsyncSessionLocal() as db:
        report = await reindex_recipes(db, embedder, force=force)
    print(f"(Re)indexed {report.indexed}, already up to date {report.unchanged}")


async def _llm_status() -> None:
    """Print the chat and embedding server status."""
    async with OllamaClient.for_chat() as chat, OllamaClient.for_embeddings() as embeddings:
        for role, client in (("chat", chat), ("embeddings", embeddings)):
            print(f"{role}: {(await client.status()).model_dump_json(indent=2)}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the chosen command; returns the process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="Recipe knowledge base tasks."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest-mealdb", help="import and index TheMealDB recipes")
    ingest.add_argument("--letters", default=ALL_LETTERS, help="first letters to fetch")
    ingest.add_argument("--limit", type=int, default=None, help="import at most N recipes")
    ingest.add_argument("--force", action="store_true", help="re-embed unchanged recipes too")

    reindex = commands.add_parser("reindex", help="embed recipes with missing/stale chunks")
    reindex.add_argument("--force", action="store_true", help="re-embed every recipe")

    commands.add_parser("llm-status", help="check the Ollama servers and models")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "ingest-mealdb":
            asyncio.run(_ingest_mealdb(args.letters, args.limit, args.force))
        elif args.command == "reindex":
            asyncio.run(_reindex(args.force))
        else:
            asyncio.run(_llm_status())
    except (LLMError, RecipeSourceError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
