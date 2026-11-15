"""Command-line tools for building the recipe knowledge base.

Usage (from `backend/`, or inside the backend container)::

    python -m app.cli import-themealdb [--letters abc] [--limit N] [--no-index]
    python -m app.cli reindex [--force]
"""

import argparse
import asyncio
import logging
import string
import sys
from collections.abc import Sequence

from app.core.config import settings
from app.core.exceptions import LLMServiceError, RecipeSourceError
from app.db.session import AsyncSessionLocal, init_db
from app.llm import ollama_client
from app.rag import reindex_recipes
from app.services import ImportResult, import_recipes
from app.sources import themealdb_client

logger = logging.getLogger("app.cli")

_PROGRESS_EVERY = 25


async def _import_themealdb(letters: str, limit: int | None, index: bool) -> None:
    """Import (and optionally index) TheMealDB meals starting with any of `letters`."""
    await init_db()

    def report(result: ImportResult) -> None:
        """Log the running totals every few recipes."""
        if result.imported % _PROGRESS_EVERY == 0:
            logger.info("... %d recipes imported, %d indexed", result.imported, result.indexed)

    async with (
        themealdb_client(settings) as mealdb,
        ollama_client(settings) as ollama,
        AsyncSessionLocal() as db,
    ):
        # Fail fast with a clear message instead of after the first download.
        if index and (missing := await ollama.missing_models()):
            raise LLMServiceError(
                f"Ollama models not pulled: {', '.join(missing)} (run `make models`)"
            )
        result = await import_recipes(
            db,
            mealdb.iter_meals(letters),
            embedder=ollama.embedder if index else None,
            limit=limit,
            on_progress=report,
        )
    logger.info(
        "Done: %d created, %d updated, %d (re)indexed",
        result.created,
        result.updated,
        result.indexed,
    )


async def _reindex(force: bool) -> None:
    """Embed every recipe whose chunks are missing or stale (all of them with `force`)."""
    await init_db()

    def report(processed: int) -> None:
        """Log progress every few recipes."""
        if processed % _PROGRESS_EVERY == 0:
            logger.info("... %d recipes processed", processed)

    async with ollama_client(settings) as ollama, AsyncSessionLocal() as db:
        result = await reindex_recipes(db, ollama.embedder, force=force, on_progress=report)
    logger.info("Done: %d (re)indexed, %d already up to date", result.indexed, result.unchanged)


def _letters(value: str) -> str:
    """Validate the `--letters` argument (ASCII letters only)."""
    letters = value.lower()
    if not letters or any(char not in string.ascii_lowercase for char in letters):
        raise argparse.ArgumentTypeError("expected ASCII letters, e.g. 'abc'")
    return letters


def _build_parser() -> argparse.ArgumentParser:
    """Define the sub-commands and their options."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli", description="Build the recipe knowledge base."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    import_cmd = commands.add_parser(
        "import-themealdb", help="import recipes from TheMealDB and index them"
    )
    import_cmd.add_argument(
        "--letters",
        type=_letters,
        default=string.ascii_lowercase,
        help="only meals whose name starts with these letters (default: all)",
    )
    import_cmd.add_argument("--limit", type=int, default=None, help="stop after N recipes")
    import_cmd.add_argument(
        "--no-index", action="store_true", help="import only; run `reindex` later"
    )

    reindex_cmd = commands.add_parser("reindex", help="(re)build the chunk embeddings")
    reindex_cmd.add_argument(
        "--force", action="store_true", help="re-embed recipes even if they are up to date"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code.

    Args:
        argv: Arguments without the program name; defaults to `sys.argv[1:]`.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Request logs of the HTTP clients would drown the progress output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "import-themealdb":
            asyncio.run(_import_themealdb(args.letters, args.limit, index=not args.no_index))
        else:
            asyncio.run(_reindex(args.force))
    except (LLMServiceError, RecipeSourceError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
