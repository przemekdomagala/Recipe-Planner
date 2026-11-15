"""Ingestion: load external recipes into the knowledge base and index them."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.schemas import ExternalRecipe
from app.services.indexing import index_recipe
from app.services.llm import Embedder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestReport:
    """Outcome of an ingestion run."""

    created: int = 0
    updated: int = 0
    indexed: int = 0


async def ingest_recipes(
    db: AsyncSession,
    recipes: Sequence[ExternalRecipe],
    embedder: Embedder,
    *,
    force_reindex: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> IngestReport:
    """Upsert `recipes` and (re)index the ones whose text changed.

    Idempotent: running it again with the same data only re-checks the recipes. Each
    recipe is committed on its own, so an interrupted run keeps what it finished.

    Args:
        db: Database session.
        recipes: Recipes read from an external source.
        embedder: Embedding model used for indexing.
        force_reindex: Re-embed every recipe, even unchanged ones.
        on_progress: Called with `(done, total)` after each recipe.

    Raises:
        LLMError: If embedding fails (recipes processed so far stay committed).
    """
    report = IngestReport()
    for done, data in enumerate(recipes, start=1):
        recipe, created = await crud.recipe.upsert_external_recipe(db, data)
        if created:
            report.created += 1
        else:
            report.updated += 1
        if await index_recipe(db, recipe, embedder, force=force_reindex):
            report.indexed += 1
        if on_progress is not None:
            on_progress(done, len(recipes))
    logger.info("Ingestion finished: %s", report)
    return report
