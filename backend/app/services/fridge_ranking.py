"""Deterministic part of the fridge suggestions: matching, adaptation and scoring.

Everything here is plain Python over recipe ingredient names, so it can be tested without
a database or an LLM. The LLM step in `app.services.fridge` only re-orders the shortlist
built here and proposes swaps, which go through `adapt` like the rule-based ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from app.models import Recipe
from app.schemas import Substitution
from app.services.dietary import DietaryProfile, check_ingredient, find_violations, is_verifiable
from app.services.substitutions import allowed_substitutes
from app.services.taxonomy import is_staple, same_ingredient


@dataclass(frozen=True, slots=True)
class FridgeEntry:
    """An ingredient at hand, with its expiry date if known."""

    name: str
    expires_on: date | None = None


@dataclass(frozen=True, slots=True)
class Fridge:
    """The usable fridge contents and which of them should be used soon."""

    entries: tuple[FridgeEntry, ...]
    expiring: frozenset[str] = frozenset()
    expired: tuple[str, ...] = ()

    @classmethod
    def build(cls, entries: Iterable[FridgeEntry], *, today: date, expiring_days: int) -> Fridge:
        """Sort entries into usable, expiring soon and already expired.

        Args:
            entries: Everything in the fridge.
            today: The reference date.
            expiring_days: Entries expiring within this many days count as expiring.

        Returns:
            A fridge without the expired entries; their names are kept in `expired`.
        """
        usable: list[FridgeEntry] = []
        expired: list[str] = []
        for entry in entries:
            if entry.expires_on is not None and entry.expires_on < today:
                expired.append(entry.name)
            else:
                usable.append(entry)
        horizon = today + timedelta(days=expiring_days)
        expiring = frozenset(
            e.name for e in usable if e.expires_on is not None and e.expires_on <= horizon
        )
        return cls(entries=tuple(usable), expiring=expiring, expired=tuple(expired))

    @property
    def names(self) -> list[str]:
        """Names of the usable entries."""
        return [entry.name for entry in self.entries]

    def find(self, ingredient: str) -> FridgeEntry | None:
        """Return the entry that provides `ingredient`, if any."""
        return next((e for e in self.entries if same_ingredient(e.name, ingredient)), None)


@dataclass(frozen=True, slots=True)
class ProposedSwap:
    """A substitution proposed by the LLM, not yet validated."""

    original: str
    replacement: str


@dataclass(slots=True)
class Adaptation:
    """The substitutions that make a recipe fit the profile, plus rejected proposals."""

    substitutions: list[Substitution]
    rejected: list[str] = field(default_factory=list)

    def effective(self, ingredients: Sequence[str]) -> list[str]:
        """Return the ingredient list with the substitutions applied."""
        swaps = {s.original: s.replacement for s in self.substitutions}
        return [swaps.get(name, name) for name in ingredients]


@dataclass(frozen=True, slots=True)
class Evaluation:
    """How well an (adapted) recipe fits the fridge."""

    used: list[str]
    missing: list[str]
    expiring_used: list[str]
    unverified: list[str]
    coverage: float
    usage: float
    expiring_share: float


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Weights of the deterministic score; they sum to 1 so the score stays in [0, 1]."""

    coverage: float = 0.45
    usage: float = 0.2
    semantic: float = 0.2
    expiring: float = 0.15


DEFAULT_WEIGHTS = RankingWeights()


@dataclass(slots=True)
class Candidate:
    """A recipe on the shortlist, with its adaptation and evaluation."""

    recipe: Recipe
    ingredients: list[str]
    semantic: float
    adaptation: Adaptation
    evaluation: Evaluation
    score: float


def recipe_ingredient_names(recipe: Recipe) -> list[str]:
    """Return the ingredient names of a recipe whose ingredients are loaded."""
    return [item.ingredient.name for item in recipe.ingredients]


def adapt(
    ingredients: Sequence[str],
    profile: DietaryProfile,
    fridge: Fridge,
    proposed: Iterable[ProposedSwap] = (),
) -> Adaptation | None:
    """Choose substitutions so that the recipe satisfies the profile.

    Proposed swaps are applied first if they pass validation: the original must be in the
    recipe, the replacement must satisfy the profile and be recognised by the taxonomy, and
    a swap must either fix a violation or use something from the fridge. Every violation
    left over gets the best allowed swap from the substitution table, preferring
    ingredients that are in the fridge.

    Args:
        ingredients: The recipe's ingredient names.
        profile: The constraints to satisfy.
        fridge: The fridge contents.
        proposed: Swaps suggested by the LLM.

    Returns:
        The adaptation, or `None` if some violation has no allowed substitute.
    """
    violations = {v.ingredient: v for v in find_violations(ingredients, profile)}
    chosen: dict[str, Substitution] = {}
    rejected: list[str] = []

    for swap in proposed:
        original = next((n for n in ingredients if same_ingredient(n, swap.original)), None)
        replacement = " ".join(swap.replacement.lower().split())
        label = f"{swap.original} -> {swap.replacement}"
        if original is None:
            rejected.append(f"{label}: not an ingredient of the recipe")
            continue
        if original in chosen or not replacement or same_ingredient(original, replacement):
            continue
        problem = check_ingredient(replacement, profile)
        if problem is not None:
            rejected.append(f"{label}: {'; '.join(problem.reasons)}")
            continue
        if not is_verifiable(replacement, profile):
            rejected.append(f"{label}: the dietary checker doesn't recognise '{replacement}'")
            continue
        if original in violations:
            reason = "; ".join(violations[original].reasons)
        elif fridge.find(replacement) is not None and fridge.find(original) is None:
            reason = "uses what's in your fridge"
        else:
            rejected.append(f"{label}: only swaps that fix a constraint or use the fridge apply")
            continue
        chosen[original] = Substitution(
            original=original, replacement=replacement, reason=reason, source="llm"
        )

    for name, violation in violations.items():
        if name in chosen:
            continue
        options = allowed_substitutes(name, profile, prefer=fridge.names)
        if not options:
            return None
        chosen[name] = Substitution(
            original=name,
            replacement=options[0],
            reason="; ".join(violation.reasons),
            source="rules",
        )

    order = {name: index for index, name in enumerate(ingredients)}
    substitutions = sorted(chosen.values(), key=lambda s: order[s.original])
    return Adaptation(substitutions=substitutions, rejected=rejected)


def evaluate(
    ingredients: Sequence[str], adaptation: Adaptation, fridge: Fridge, profile: DietaryProfile
) -> Evaluation:
    """Measure how much of the adapted recipe the fridge covers.

    Staples (salt, pepper, oil, water) are assumed to be at hand and don't count.

    Args:
        ingredients: The recipe's ingredient names.
        adaptation: The substitutions to apply first.
        fridge: The fridge contents.
        profile: Used to flag ingredients the checker can't verify.

    Returns:
        The evaluation; `coverage` is the share of non-staple ingredients in the fridge,
        `usage` the share of fridge items used, `expiring_share` the share of expiring
        items used.
    """
    used: list[str] = []
    missing: list[str] = []
    entries_used: set[str] = set()
    effective = list(dict.fromkeys(adaptation.effective(ingredients)))
    for name in effective:
        if is_staple(name):
            continue
        entry = fridge.find(name)
        if entry is None:
            missing.append(name)
        else:
            used.append(name)
            entries_used.add(entry.name)
    needed = len(used) + len(missing)
    expiring_used = sorted(entries_used & fridge.expiring)
    return Evaluation(
        used=used,
        missing=missing,
        expiring_used=expiring_used,
        unverified=[name for name in effective if not is_verifiable(name, profile)],
        coverage=len(used) / needed if needed else 0.0,
        usage=len(entries_used) / len(fridge.entries) if fridge.entries else 0.0,
        expiring_share=len(expiring_used) / len(fridge.expiring) if fridge.expiring else 0.0,
    )


def score(
    evaluation: Evaluation, semantic: float, weights: RankingWeights = DEFAULT_WEIGHTS
) -> float:
    """Combine an evaluation and the semantic similarity into a score in [0, 1]."""
    total = (
        weights.coverage * evaluation.coverage
        + weights.usage * evaluation.usage
        + weights.semantic * min(max(semantic, 0.0), 1.0)
        + weights.expiring * evaluation.expiring_share
    )
    return round(total, 4)


def build_candidate(
    recipe: Recipe,
    fridge: Fridge,
    profile: DietaryProfile,
    *,
    semantic: float = 0.0,
    proposed: Iterable[ProposedSwap] = (),
) -> Candidate | None:
    """Adapt, evaluate and score one recipe.

    Args:
        recipe: Recipe with its ingredients loaded.
        fridge: The fridge contents.
        profile: The constraints to satisfy.
        semantic: Similarity of the recipe to the search query, if it was retrieved.
        proposed: Swaps suggested by the LLM.

    Returns:
        The candidate, or `None` if the recipe can't be adapted to the profile.
    """
    ingredients = recipe_ingredient_names(recipe)
    adaptation = adapt(ingredients, profile, fridge, proposed)
    if adaptation is None:
        return None
    evaluation = evaluate(ingredients, adaptation, fridge, profile)
    return Candidate(
        recipe=recipe,
        ingredients=ingredients,
        semantic=semantic,
        adaptation=adaptation,
        evaluation=evaluation,
        score=score(evaluation, semantic),
    )


def shortlist(
    recipes: Iterable[Recipe],
    fridge: Fridge,
    profile: DietaryProfile,
    *,
    semantic: dict[int, float],
    max_missing: int,
) -> list[Candidate]:
    """Return the recipes that can be made from the fridge, best score first.

    A recipe qualifies if it can be adapted to the profile, uses at least one fridge item
    and needs at most `max_missing` other ingredients.

    Args:
        recipes: Candidate recipes with their ingredients loaded.
        fridge: The fridge contents.
        profile: The constraints to satisfy.
        semantic: Semantic similarity by recipe id, for recipes found by vector search.
        max_missing: Most ingredients the user would have to buy.
    """
    candidates = []
    for recipe in recipes:
        candidate = build_candidate(recipe, fridge, profile, semantic=semantic.get(recipe.id, 0.0))
        if (
            candidate is not None
            and candidate.evaluation.used
            and len(candidate.evaluation.missing) <= max_missing
        ):
            candidates.append(candidate)
    return sorted(candidates, key=lambda c: (-c.score, c.recipe.id))
