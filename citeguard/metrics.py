"""검색 평가 지표.

라벨셋 기반 정보검색(IR) 표준 지표: Hit@K, Reciprocal Rank, MRR.
운영 파이프라인의 "Top-K 안에 정답이 있는가"를 그대로 계량화한 것으로,
실서비스 평가 체계와 지표 정의를 일치시키는 것이 핵심이다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> bool:
    """상위 k개 후보 안에 정답이 하나라도 있으면 True."""
    if k <= 0:
        raise ValueError("k must be positive")
    return any(candidate in gold for candidate in ranked[:k])


def reciprocal_rank(ranked: Sequence[str], gold: set[str]) -> float:
    """첫 정답의 역순위(1/rank). 정답이 없으면 0."""
    for i, candidate in enumerate(ranked, start=1):
        if candidate in gold:
            return 1.0 / i
    return 0.0


def mrr(reciprocal_ranks: Iterable[float]) -> float:
    """Mean Reciprocal Rank — 질의별 역순위의 평균."""
    ranks = list(reciprocal_ranks)
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)
