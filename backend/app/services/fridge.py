"""'What's in the Fridge': recipe suggestions from the RAG pipeline, ranked by the LLM.

Pipeline:

1. **Retrieve** a candidate pool with `recipe_search`, querying with the fridge contents
   (phrased like the recipes' overview chunks).
2. **Analyze** each candidate deterministically: which ingredients the user has, which are
   missing, and which break the restrictions (`app.services.dietary`). Candidates with
   too many conflicts are dropped; the best of the rest go into the prompt.
3. **Generate**: the chat model picks and orders recipes and proposes adaptations
   (substitutions for conflicting or missing ingredients), as JSON matching a schema.
4. **Post-process**, which is where restrictions are enforced: picks of unknown recipes and
   adaptations of ingredients the recipe doesn't have are discarded, unsafe substitutes are
   rejected, and the adapted ingredient list is re-checked; a recipe that still conflicts
   is excluded. Free slots are filled with the best compliant candidates.

If the model's reply isn't valid JSON for the schema, the suggestions fall back to the
deterministic ranking, so a small or misbehaving model degrades quality, never safety.
"""

import logging
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app import crud
from app.core.config import settings
from app.models import Recipe
from app.schemas import (
    Adaptation,
    ExcludedRecipe,
    FridgeRequest,
    FridgeResponse,
    FridgeSuggestion,
    RecipeSearchHit,
    RecipeSearchRequest,
    normalize_ingredient_name,
)
from app.services.dietary import DietaryConstraints, Violation
from app.services.ingredient_text import find_ingredient
from app.services.llm import ChatMessage, ChatModel, Embedder
from app.services.recipe_search import search_recipes

logger = logging.getLogger(__name__)

# Assumed to be in every kitchen when `assume_staples` is set (exact names only, so that
# "red pepper" or "sesame oil" still count as real ingredients).
PANTRY_STAPLES = frozenset(
    {
        "salt",
        "sea salt",
        "pepper",
        "black pepper",
        "salt and pepper",
        "water",
        "oil",
        "olive oil",
        "vegetable oil",
        "sunflower oil",
        "sugar",
    }
)
# Ranking weights: covering the recipe matters more than how similar its text is.
_COVERAGE_WEIGHT = 0.7
_SIMILARITY_WEIGHT = 0.3
_CONFLICT_PENALTY = 0.15
# Keep the prompt short: on CPU, prompt processing dominates the time to an answer.
_MAX_METHOD_CHARS = 300
# Length caps for the model's reply. They go into the JSON schema, which Ollama turns into
# a grammar, so small models can't ramble past the token limit and truncate the JSON.
_MAX_REASON_CHARS = 200
_MAX_NOTE_CHARS = 120
_MAX_NAME_CHARS = 60
_MAX_ADAPTATIONS = 3

SYSTEM_PROMPT = """\
You are the cooking assistant of a meal-planning app. The user lists what is in their fridge.
Choose the best recipes for them from the numbered candidates ONLY, best first.
- Prefer recipes that use more of the fridge and need fewer missing ingredients.
- A candidate's "Conflicts" are ingredients the user must not eat (allergy, diet or dislike).
  Choose such a recipe only if every conflict can be replaced or left out, and add one
  adaptation per conflict: "replace" = the conflicting ingredient, "substitute" = a safe
  alternative, or null to leave it out.
- You may add adaptations that swap a missing ingredient for something from the fridge.
- Never use a substitute that breaks the user's restrictions.
- Keep adaptations to those two kinds, and never "replace" an ingredient with itself.
- "reason": one short sentence on why the recipe fits. "note": a short how-to.
- Write reasons and notes in the language of the user's request (English if there is none)."""


# The length caps live only in the JSON schema (`json_schema_extra`), where they shape
# generation; validation stays lenient and over-long values from a model that ignores the
# schema are truncated in post-processing instead of failing the whole reply.


class _LLMAdaptation(BaseModel):
    """An adaptation as proposed by the model (validated before use)."""

    replace: str | None = Field(default=None, json_schema_extra={"maxLength": _MAX_NAME_CHARS})
    substitute: str | None = Field(default=None, json_schema_extra={"maxLength": _MAX_NAME_CHARS})
    note: str = Field(default="", json_schema_extra={"maxLength": _MAX_NOTE_CHARS})


class _LLMPick(BaseModel):
    """One recipe chosen by the model, by its number in the prompt."""

    recipe: int
    reason: str = Field(default="", json_schema_extra={"maxLength": _MAX_REASON_CHARS})
    adaptations: list[_LLMAdaptation] = Field(
        default_factory=list, json_schema_extra={"maxItems": _MAX_ADAPTATIONS}
    )


class _LLMReply(BaseModel):
    """The structured reply the model is asked for."""

    suggestions: list[_LLMPick]


_BASE_SCHEMA = _LLMReply.model_json_schema()


def reply_schema(candidates: int, max_suggestions: int) -> dict[str, Any]:
    """JSON schema for the reply: valid recipe numbers only, at most `max_suggestions`."""
    schema = deepcopy(_BASE_SCHEMA)
    schema["properties"]["suggestions"]["maxItems"] = max_suggestions
    schema["$defs"]["_LLMPick"]["properties"]["recipe"]["enum"] = list(range(1, candidates + 1))
    return schema


@dataclass(frozen=True, slots=True)
class _Coverage:
    """How a list of recipe ingredients relates to the fridge."""

    used: list[str]
    missing: list[str]
    share: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved recipe, analyzed against the fridge and the restrictions."""

    recipe: Recipe
    hit: RecipeSearchHit
    ingredients: list[str]
    coverage: _Coverage
    conflicts: list[Violation]

    @property
    def score(self) -> float:
        """Deterministic rank: coverage and similarity, minus a penalty per conflict."""
        return (
            _COVERAGE_WEIGHT * self.coverage.share
            + _SIMILARITY_WEIGHT * max(self.hit.score, 0.0)
            - _CONFLICT_PENALTY * len(self.conflicts)
        )


@dataclass(frozen=True, slots=True)
class _Adapted:
    """A candidate after applying the model's adaptations."""

    adaptations: list[Adaptation]
    coverage: _Coverage
    conflicts: list[Violation]


class _Kitchen:
    """The user's fridge, staples and restrictions; analyzes ingredient lists against them."""

    def __init__(self, request: FridgeRequest) -> None:
        """Capture the fridge contents and restrictions of `request`."""
        self.fridge = request.ingredients
        self.constraints = DietaryConstraints.from_request(request)
        self._staples = PANTRY_STAPLES if request.assume_staples else frozenset()

    def has(self, ingredient: str) -> bool:
        """Whether the user has `ingredient` (in the fridge or as a staple)."""
        if normalize_ingredient_name(ingredient) in self._staples:
            return True
        return find_ingredient(self.fridge, ingredient) is not None

    def coverage(self, ingredients: Sequence[str]) -> _Coverage:
        """Split ingredients into used and missing; staples don't count toward the share."""
        used = [name for name in ingredients if self.has(name)]
        missing = [name for name in ingredients if not self.has(name)]
        counted = [
            name for name in ingredients if normalize_ingredient_name(name) not in self._staples
        ]
        share = sum(1 for name in counted if self.has(name)) / len(counted) if counted else 0.0
        return _Coverage(used=used, missing=missing, share=round(share, 3))

    def analyze(self, recipe: Recipe, hit: RecipeSearchHit) -> Candidate:
        """Analyze a retrieved recipe (ingredients loaded)."""
        names = [item.ingredient.name for item in recipe.ingredients]
        return Candidate(
            recipe=recipe,
            hit=hit,
            ingredients=names,
            coverage=self.coverage(names),
            conflicts=self.constraints.violations(names),
        )

    def adapt(self, candidate: Candidate, proposed: Sequence[_LLMAdaptation]) -> _Adapted:
        """Apply the model's adaptations that are valid and safe, then re-check the recipe.

        An adaptation is discarded if it replaces an ingredient the recipe doesn't have
        (hallucinated) or if its substitute breaks the restrictions.
        """
        ingredients = list(candidate.ingredients)
        kept: list[Adaptation] = []
        for item in proposed:
            target = _clean(item.replace)
            if target is not None:
                target = find_ingredient(ingredients, target)
                if target is None:
                    continue
            substitute = _clean(item.substitute)
            if target is not None and substitute == target:
                substitute, target = None, None  # A no-op swap; keep it only as a tip.
            if substitute is not None and self.constraints.reasons(substitute):
                continue
            if target is None and substitute is None and not item.note.strip():
                continue
            if target is not None:
                ingredients.remove(target)
            if substitute is not None and find_ingredient(ingredients, substitute) is None:
                ingredients.append(substitute)
            note = item.note.strip()[:_MAX_NOTE_CHARS]
            kept.append(Adaptation(replace=target, substitute=substitute, note=note))
        return _Adapted(
            adaptations=kept,
            coverage=self.coverage(ingredients),
            conflicts=self.constraints.violations(ingredients),
        )


def _clean(value: str | None) -> str | None:
    """Normalize an ingredient name from the model; blank means none."""
    return normalize_ingredient_name(value or "")[:_MAX_NAME_CHARS] or None


def build_query(request: FridgeRequest) -> str:
    """Retrieval query phrased like a recipe overview chunk, plus diet and wish."""
    diet = f"{request.diet.value.capitalize()} recipe" if request.diet else "Recipe"
    query = f"{diet} with ingredients: {', '.join(request.ingredients)}"
    return f"{query}\n{request.note}" if request.note else query


def build_messages(
    request: FridgeRequest, constraints: DietaryConstraints, candidates: Sequence[Candidate]
) -> list[ChatMessage]:
    """Assemble the system prompt and the fridge, restrictions and numbered candidates."""
    blocks = []
    for number, candidate in enumerate(candidates, start=1):
        recipe = candidate.recipe
        meta = ", ".join(part for part in (recipe.category, recipe.area) if part)
        method = " ".join(recipe.instructions.split())
        if len(method) > _MAX_METHOD_CHARS:
            method = method[:_MAX_METHOD_CHARS].rstrip() + " [...]"
        conflicts = "; ".join(map(str, candidate.conflicts)) or "none"
        blocks.append(
            f"[{number}] {recipe.title}" + (f" ({meta})" if meta else "") + "\n"
            f"Have: {', '.join(candidate.coverage.used) or 'nothing'}\n"
            f"Missing: {', '.join(candidate.coverage.missing) or 'nothing'}\n"
            f"Conflicts: {conflicts}\n"
            f"Method: {method}"
        )
    user_prompt = (
        f"Fridge: {', '.join(request.ingredients)}\n"
        f"Restrictions: {constraints.describe()}\n"
        + (f"Request: {request.note}\n" if request.note else "")
        + f"Choose up to {request.max_suggestions} recipes.\n\n"
        "Candidates:\n\n" + "\n\n".join(blocks)
    )
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user_prompt)]


async def _ask_model(
    chat_model: ChatModel, messages: Sequence[ChatMessage], schema: dict[str, Any]
) -> _LLMReply | None:
    """Ask for structured picks; `None` if the reply doesn't match the schema.

    Raises:
        LLMError: If the model can't be reached (unlike a bad reply, this isn't recoverable).
    """
    raw = await chat_model.chat(messages, json_schema=schema, max_tokens=settings.fridge_max_tokens)
    try:
        return _LLMReply.model_validate_json(raw)
    except ValidationError:
        logger.warning("Fridge suggestions: invalid structured reply, using fallback ranking")
        return None


def _suggestion(
    rank: int,
    candidate: Candidate,
    *,
    reason: str | None,
    adaptations: list[Adaptation],
    coverage: _Coverage,
) -> FridgeSuggestion:
    """Build the API representation of a suggestion."""
    recipe = candidate.recipe
    return FridgeSuggestion(
        rank=rank,
        recipe_id=recipe.id,
        title=recipe.title,
        category=recipe.category,
        area=recipe.area,
        thumbnail_url=recipe.thumbnail_url,
        reason=reason,
        used_ingredients=coverage.used,
        missing_ingredients=coverage.missing,
        adaptations=adaptations,
        coverage=coverage.share,
        similarity=candidate.hit.score,
    )


async def suggest_recipes(
    db: AsyncSession, request: FridgeRequest, embedder: Embedder, chat_model: ChatModel
) -> FridgeResponse:
    """Suggest recipes for the fridge contents that satisfy every restriction.

    The chat model isn't called when no candidate survives retrieval and pre-filtering.

    Raises:
        LLMError: If embedding fails or the chat model is unavailable.
    """
    kitchen = _Kitchen(request)
    search = await search_recipes(
        db,
        RecipeSearchRequest(
            query=build_query(request),
            limit=settings.fridge_candidate_pool,
            filters=request.filters,
        ),
        embedder,
    )
    hits = {hit.recipe_id: hit for hit in search.results}
    recipes = await crud.recipe.get_recipes_by_ids(db, list(hits))
    candidates = sorted(
        (kitchen.analyze(recipe, hits[recipe.id]) for recipe in recipes),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    # Recipes that can't be suggested, with the reasons (updated after post-processing).
    blocked: dict[int, list[Violation]] = {
        c.recipe.id: c.conflicts for c in candidates if c.conflicts
    }
    context = [c for c in candidates if len(c.conflicts) <= settings.fridge_max_conflicts][
        : settings.fridge_context_recipes
    ]

    suggestions: list[FridgeSuggestion] = []
    model: str | None = None
    if context:
        reply = await _ask_model(
            chat_model,
            build_messages(request, kitchen.constraints, context),
            reply_schema(len(context), request.max_suggestions),
        )
        if reply is not None:
            model = chat_model.chat_model
            for pick in reply.suggestions:
                if len(suggestions) >= request.max_suggestions:
                    break
                if not 1 <= pick.recipe <= len(context):
                    continue
                candidate = context[pick.recipe - 1]
                if any(s.recipe_id == candidate.recipe.id for s in suggestions):
                    continue
                adapted = kitchen.adapt(candidate, pick.adaptations)
                if adapted.conflicts:
                    blocked[candidate.recipe.id] = adapted.conflicts
                    continue
                blocked.pop(candidate.recipe.id, None)
                suggestions.append(
                    _suggestion(
                        len(suggestions) + 1,
                        candidate,
                        reason=pick.reason.strip()[:_MAX_REASON_CHARS] or None,
                        adaptations=adapted.adaptations,
                        coverage=adapted.coverage,
                    )
                )

    # Fill free slots (or everything, without a usable reply) with compliant candidates.
    chosen = {s.recipe_id for s in suggestions}
    for candidate in context:
        if len(suggestions) >= request.max_suggestions:
            break
        if candidate.conflicts or candidate.recipe.id in chosen:
            continue
        suggestions.append(
            _suggestion(
                len(suggestions) + 1,
                candidate,
                reason=None,
                adaptations=[],
                coverage=candidate.coverage,
            )
        )

    excluded = [
        ExcludedRecipe(
            recipe_id=c.recipe.id,
            title=c.recipe.title,
            reasons=list(dict.fromkeys(map(str, blocked[c.recipe.id]))),
        )
        for c in candidates
        if c.recipe.id in blocked
    ]
    return FridgeResponse(
        ingredients=request.ingredients, suggestions=suggestions, excluded=excluded, model=model
    )
