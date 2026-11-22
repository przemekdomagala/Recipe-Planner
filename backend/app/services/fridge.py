"""Fridge suggestions ("what's in the fridge"): RAG retrieval, LLM ranking, rule validation.

Flow:

1. **Retrieve** candidates two ways: recipes sharing ingredients with the fridge (SQL) and
   recipes semantically close to the fridge contents plus the user's wish (vector search).
2. **Shortlist** them deterministically (`fridge_ranking`): adapt each recipe to the
   dietary profile with the substitution table, drop the ones that can't be adapted or
   need too many missing ingredients, and score the rest.
3. **Rank and adapt with the LLM**: the best few go into the prompt, annotated with what's
   in the fridge and what's forbidden; the model returns a JSON ranking with swaps.
4. **Validate**: unknown recipe numbers are ignored, every proposed swap is re-checked
   against the profile, leftover violations get rule-based swaps, and the fridge coverage
   is recomputed. If the LLM fails, the deterministic ranking is returned instead.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.core.exceptions import LLMError
from app.schemas import (
    FridgeSuggestion,
    RecipeSearchRequest,
    SuggestionRequest,
    SuggestionResponse,
)
from app.services.dietary import DietaryProfile, check_ingredient
from app.services.fridge_ranking import (
    Candidate,
    Fridge,
    FridgeEntry,
    ProposedSwap,
    build_candidate,
    shortlist,
)
from app.services.llm import ChatMessage, ChatModel, Embedder
from app.services.recipe_search import search_recipes
from app.services.substitutions import allowed_substitutes
from app.services.taxonomy import is_staple

logger = logging.getLogger(__name__)

EMPTY_FRIDGE_NOTE = "Your fridge is empty: add some ingredients first."
NO_MATCH_NOTE = (
    "No recipe fits your fridge and dietary profile. Add more ingredients or allow more "
    "missing ingredients."
)
_SEMANTIC_HITS = 10
_INSTRUCTION_PREVIEW_CHARS = 160

SYSTEM_PROMPT = """\
You are the "What's in the fridge" assistant of a meal-planning app.
The app has pre-selected candidate recipes that can be cooked with the user's fridge and
adapted to their dietary constraints. Choose the best ones for the user, best first.
- Use only the candidate numbers given; never invent recipes.
- If the user has a wish, it matters most. Otherwise prefer recipes that use ingredients
  marked "use soon" and need few missing ingredients.
- An ingredient marked FORBIDDEN must be swapped: pick one of its allowed swaps.
- You may also swap a missing ingredient for a similar ingredient from the fridge.
- Give each pick a one-sentence reason addressed to the user, in the language of the wish
  (English if there is none).
Reply with JSON only."""

REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipe": {"type": "integer"},
                    "reason": {"type": "string"},
                    "substitutions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string"},
                                "replacement": {"type": "string"},
                            },
                            "required": ["original", "replacement"],
                        },
                    },
                },
                "required": ["recipe", "reason", "substitutions"],
            },
        }
    },
    "required": ["suggestions"],
}


class _LLMSwap(BaseModel):
    """A swap as written by the LLM."""

    original: str
    replacement: str


class _LLMPick(BaseModel):
    """One ranked recipe as written by the LLM."""

    recipe: int
    reason: str = ""
    substitutions: list[_LLMSwap] = []


class _LLMReply(BaseModel):
    """The LLM's whole reply (`REPLY_SCHEMA`)."""

    suggestions: list[_LLMPick]


async def suggest_recipes(
    db: AsyncSession,
    request: SuggestionRequest,
    embedder: Embedder,
    chat_model: ChatModel,
    *,
    today: date | None = None,
) -> SuggestionResponse:
    """Suggest recipes for the fridge contents that respect the dietary profile.

    LLM failures don't fail the request: without embeddings only the ingredient-overlap
    retrieval is used, and without the chat model the deterministic ranking is returned.
    Both cases are reported in `notes`.

    Args:
        db: Database session.
        request: Options; the stored fridge and profile are used unless overridden.
        embedder: Embedding model for the semantic retrieval.
        chat_model: Chat model that ranks and adapts the shortlist.
        today: Reference date for expiry (defaults to today).

    Returns:
        The suggestions, the model that ranked them and notes on fallbacks.
    """
    notes: list[str] = []
    profile = await _resolve_profile(db, request)
    fridge = await _resolve_fridge(db, request, today or date.today())
    if fridge.expired:
        notes.append("Skipped expired items: " + ", ".join(fridge.expired) + ".")
    if not fridge.entries:
        return SuggestionResponse(
            suggestions=[], model=None, llm_used=False, notes=[*notes, EMPTY_FRIDGE_NOTE]
        )

    semantic = await _semantic_scores(db, request, fridge, embedder, notes)
    overlap_ids = await _overlap_recipe_ids(db, fridge)
    candidate_ids = list(dict.fromkeys([*overlap_ids, *semantic]))
    recipes = await crud.recipe.get_recipes_by_ids(db, candidate_ids)
    candidates = shortlist(
        recipes, fridge, profile, semantic=semantic, max_missing=request.max_missing
    )
    if not candidates:
        return SuggestionResponse(
            suggestions=[], model=None, llm_used=False, notes=[*notes, NO_MATCH_NOTE]
        )

    if request.use_llm:
        pool = candidates[: max(request.limit, settings.fridge_llm_candidates)]
        try:
            suggestions = await _rank_with_llm(pool, fridge, profile, request, chat_model, notes)
        except LLMError as exc:
            logger.warning("Fridge ranking without the LLM: %s", exc)
            notes.append(f"AI ranking unavailable ({exc}); showing the rule-based ranking.")
        except (ValidationError, ValueError) as exc:
            logger.warning("Unparseable LLM fridge ranking: %s", exc)
            notes.append("The AI's answer couldn't be parsed; showing the rule-based ranking.")
        else:
            return SuggestionResponse(
                suggestions=suggestions, model=chat_model.chat_model, llm_used=True, notes=notes
            )

    suggestions = [_to_suggestion(c, reason=None) for c in candidates[: request.limit]]
    return SuggestionResponse(suggestions=suggestions, model=None, llm_used=False, notes=notes)


async def _resolve_profile(db: AsyncSession, request: SuggestionRequest) -> DietaryProfile:
    """Return the request's profile, or the stored one."""
    if request.profile is not None:
        data = request.profile
        return DietaryProfile.from_values(data.diets, data.allergens, data.excluded_ingredients)
    stored = await crud.profile.get_profile(db)
    return DietaryProfile.from_values(stored.diets, stored.allergens, stored.excluded_ingredients)


async def _resolve_fridge(db: AsyncSession, request: SuggestionRequest, today: date) -> Fridge:
    """Return the request's ingredients, or the stored fridge items, as a `Fridge`."""
    if request.ingredients is not None:
        entries = [FridgeEntry(name) for name in request.ingredients]
    else:
        items = await crud.fridge.get_fridge_items(db)
        entries = [FridgeEntry(item.ingredient.name, item.expires_on) for item in items]
    return Fridge.build(entries, today=today, expiring_days=settings.fridge_expiring_days)


async def _overlap_recipe_ids(db: AsyncSession, fridge: Fridge) -> list[int]:
    """Return the recipes sharing the most (non-staple) ingredients with the fridge."""
    ingredient_ids = [
        ingredient.id
        for ingredient in await crud.ingredient.get_all_ingredients(db)
        if not is_staple(ingredient.name) and fridge.find(ingredient.name) is not None
    ]
    return await crud.recipe.find_recipe_ids_by_ingredients(
        db, ingredient_ids, limit=settings.fridge_candidate_pool
    )


async def _semantic_scores(
    db: AsyncSession,
    request: SuggestionRequest,
    fridge: Fridge,
    embedder: Embedder,
    notes: list[str],
) -> dict[int, float]:
    """Run the vector search for the fridge and the wish; returns scores by recipe id."""
    query = "Recipe with " + ", ".join(fridge.names)
    if request.query:
        query = f"{request.query}. {query}"
    try:
        search = await search_recipes(
            db, RecipeSearchRequest(query=query[:500], limit=_SEMANTIC_HITS), embedder
        )
    except LLMError as exc:
        logger.warning("Fridge retrieval without embeddings: %s", exc)
        notes.append(f"Semantic search unavailable ({exc}); matched by ingredients only.")
        return {}
    return {hit.recipe_id: hit.score for hit in search.results}


def build_messages(
    pool: list[Candidate], fridge: Fridge, profile: DietaryProfile, request: SuggestionRequest
) -> list[ChatMessage]:
    """Assemble the prompt: fridge, constraints, wish and the annotated candidates."""
    fridge_lines = ", ".join(
        f"{name} (use soon)" if name in fridge.expiring else name for name in fridge.names
    )
    blocks = []
    for number, candidate in enumerate(pool, start=1):
        recipe = candidate.recipe
        meta = ", ".join(part for part in (recipe.area, recipe.category) if part)
        evaluation = candidate.evaluation
        have = len(evaluation.used)
        lines = [
            f"[{number}] {recipe.title}"
            + (f" ({meta})" if meta else "")
            + f" - {have} of {have + len(evaluation.missing)} ingredients at hand"
        ]
        lines += [f"- {_annotate(name, fridge, profile)}" for name in candidate.ingredients]
        if request.query:
            preview = " ".join(recipe.instructions.split())[:_INSTRUCTION_PREVIEW_CHARS]
            lines.append(f"Method: {preview}...")
        blocks.append("\n".join(lines))
    user_prompt = (
        f"Fridge: {fridge_lines}\n"
        f"Dietary constraints: {profile.describe()}\n"
        f"Wish: {request.query or 'none'}\n"
        f"Pick up to {request.limit} recipes.\n\n"
        "Candidates:\n\n" + "\n\n".join(blocks)
    )
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user_prompt)]


def _annotate(name: str, fridge: Fridge, profile: DietaryProfile) -> str:
    """Describe one ingredient for the prompt: in the fridge, missing, staple or forbidden."""
    violation = check_ingredient(name, profile)
    if violation is not None:
        swaps = allowed_substitutes(name, profile, prefer=fridge.names)
        return (
            f"{name}: FORBIDDEN ({'; '.join(violation.reasons)}); allowed swaps: {', '.join(swaps)}"
        )
    if is_staple(name):
        return f"{name}: staple"
    entry = fridge.find(name)
    if entry is None:
        return f"{name}: missing"
    return f"{name}: in fridge" + (", use soon" if entry.name in fridge.expiring else "")


def parse_reply(text: str) -> _LLMReply:
    """Parse the LLM's JSON reply, tolerating a surrounding Markdown code fence.

    Raises:
        ValidationError: If the reply isn't JSON matching `REPLY_SCHEMA`.
    """
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return _LLMReply.model_validate_json(match.group(1) if match else text)


async def _rank_with_llm(
    pool: list[Candidate],
    fridge: Fridge,
    profile: DietaryProfile,
    request: SuggestionRequest,
    chat_model: ChatModel,
    notes: list[str],
) -> list[FridgeSuggestion]:
    """Let the LLM rank and adapt the shortlist, then validate its answer.

    Raises:
        LLMError: If the chat model fails.
        ValidationError: If its reply can't be parsed.
    """
    reply = parse_reply(
        await chat_model.chat(
            build_messages(pool, fridge, profile, request), json_schema=REPLY_SCHEMA
        )
    )
    picked: list[FridgeSuggestion] = []
    seen: set[int] = set()
    for pick in reply.suggestions:
        if len(picked) == request.limit:
            break
        if not 1 <= pick.recipe <= len(pool):
            notes.append(f"Ignored the AI's reference to unknown recipe [{pick.recipe}].")
            continue
        if pick.recipe in seen:
            continue
        seen.add(pick.recipe)
        base = pool[pick.recipe - 1]
        swaps = [ProposedSwap(s.original, s.replacement) for s in pick.substitutions]
        adapted = build_candidate(
            base.recipe, fridge, profile, semantic=base.semantic, proposed=swaps
        )
        if adapted is None or len(adapted.evaluation.missing) > request.max_missing:
            adapted = base
        else:
            notes += [
                f"{base.recipe.title}: rejected swap {r}." for r in adapted.adaptation.rejected
            ]
        picked.append(_to_suggestion(adapted, reason=pick.reason.strip() or None))

    for index, candidate in enumerate(pool, start=1):
        if len(picked) == request.limit:
            break
        if index not in seen:
            picked.append(_to_suggestion(candidate, reason=None))
    return picked


def _to_suggestion(candidate: Candidate, *, reason: str | None) -> FridgeSuggestion:
    """Convert a candidate into the API representation."""
    recipe = candidate.recipe
    evaluation = candidate.evaluation
    return FridgeSuggestion(
        recipe_id=recipe.id,
        title=recipe.title,
        category=recipe.category,
        area=recipe.area,
        thumbnail_url=recipe.thumbnail_url,
        score=candidate.score,
        coverage=round(evaluation.coverage, 3),
        used_ingredients=evaluation.used,
        missing_ingredients=evaluation.missing,
        expiring_used=evaluation.expiring_used,
        substitutions=candidate.adaptation.substitutions,
        unverified_ingredients=evaluation.unverified,
        reason=reason,
    )
