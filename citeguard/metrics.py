"""Retrieval evaluation metrics.

Standard labeled-set IR metrics: Hit@K, Reciprocal Rank, MRR.
These quantify exactly "is the right answer inside the Top-K the user
sees" — keeping the evaluation aligned with the production pipeline's
definition of success.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> bool:
    """True if any gold answer appears in the top k candidates."""
    if k <= 0:
        raise ValueError("k must be positive")
    return any(candidate in gold for candidate in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    """Reciprocal rank (1/rank) of the first gold answer; 0 if absent."""
    for i, candidate in enumerate(ranked, start=1):
        if candidate in gold:
            return 1.0 / i
    return 0.0


def mrr(reciprocal_ranks: Iterable[float]) -> float:
    """Mean Reciprocal Rank — average of per-query reciprocal ranks."""
    ranks = list(reciprocal_ranks)
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)
