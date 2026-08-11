"""검색 평가 하네스 (Retrieval Eval Harness).

라벨셋(질의 → 정답 상품/문서 ID)을 검색 파이프라인에 흘려보내
Hit@K·MRR을 집계하고, 실패 질의는 **어느 단계에서 정답이 유실됐는지**
(stage-level error diagnosis)까지 자동 진단한다.

파이프라인은 단계별 후보 목록을 담은 StageTrace를 반환하기만 하면 되므로
검색 엔진 구현과 완전히 분리된다(엔진 교체 가능 구조).

단계별 진단 로직:
  후보군은 단계를 거치며 좁혀지는 깔때기(funnel) 구조라고 가정한다.
  정답이 처음으로 사라진 단계 = 오류가 발생한 단계.
  예) retrieve에는 있었는데 rerank에 없다 → 재랭킹 단계의 문제.
      retrieve부터 없다 → 검색(리콜) 단계의 문제.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .metrics import hit_at_k, mrr, reciprocal_rank


@dataclass
class StageTrace:
    """질의 1건이 파이프라인을 통과한 궤적.

    stages: (단계 이름, 그 단계의 후보 ID 목록) — 파이프라인 순서대로.
            마지막 단계가 최종 랭킹 결과다.
    meta:   디버깅용 부가 정보 (예: 파싱 결과, 채널별 후보).
            병렬 채널(alias/BM25 등)은 깔때기 단계가 아니므로
            stages가 아닌 meta에 기록한다.
    """

    stages: list[tuple[str, list[str]]]
    meta: dict = field(default_factory=dict)

    @property
    def final(self) -> list[str]:
        """최종 랭킹 결과 (마지막 단계의 후보 목록)."""
        if not self.stages:
            return []
        return self.stages[-1][1]


@dataclass(frozen=True)
class LabeledQuery:
    """라벨셋의 1행: 실제 현장 질의와 정답 ID 집합."""

    query: str
    gold: frozenset[str]
    note: str = ""


def load_labelset(path: str | Path) -> list[LabeledQuery]:
    """[{"query": "...", "gold": ["P001"], "note": "..."}] 형태의 JSON 로드."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    labelset = []
    for row in data:
        if not row.get("query") or not row.get("gold"):
            raise ValueError(f"labelset row needs query/gold: {row}")
        labelset.append(
            LabeledQuery(
                query=row["query"],
                gold=frozenset(row["gold"]),
                note=row.get("note", ""),
            )
        )
    return labelset


@dataclass
class QueryResult:
    """질의 1건의 평가 결과."""

    query: str
    gold: frozenset[str]
    trace: StageTrace
    hits: dict[int, bool]        # k → Hit@K 여부
    rr: float                    # reciprocal rank
    lost_at: str | None          # 정답이 유실된 단계 이름 (성공 시 None)
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.lost_at is None


@dataclass
class EvalReport:
    """라벨셋 전체에 대한 평가 리포트."""

    results: list[QueryResult]
    ks: tuple[int, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    def hit_rate(self, k: int) -> float:
        if not self.results:
            return 0.0
        return sum(r.hits[k] for r in self.results) / self.total

    @property
    def mrr(self) -> float:
        return mrr(r.rr for r in self.results)

    @property
    def failures(self) -> list[QueryResult]:
        return [r for r in self.results if not r.passed]

    def failures_by_stage(self) -> dict[str, list[QueryResult]]:
        """실패 질의를 유실 단계별로 그룹핑 — 개선 우선순위 판단의 근거."""
        grouped: dict[str, list[QueryResult]] = {}
        for r in self.failures:
            grouped.setdefault(r.lost_at or "unknown", []).append(r)
        return grouped

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "metrics": {
                **{f"hit@{k}": round(self.hit_rate(k), 4) for k in self.ks},
                "mrr": round(self.mrr, 4),
            },
            "failures_by_stage": {
                stage: [r.query for r in rows]
                for stage, rows in self.failures_by_stage().items()
            },
            "results": [
                {
                    "query": r.query,
                    "gold": sorted(r.gold),
                    "final_top5": r.trace.final[:5],
                    "hits": {f"hit@{k}": v for k, v in r.hits.items()},
                    "rr": round(r.rr, 4),
                    "lost_at": r.lost_at,
                    "note": r.note,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Retrieval Eval Report",
            "",
            f"- Labelset: **{self.total} queries**",
            "- " + " · ".join(f"**Hit@{k} {self.hit_rate(k):.1%}**" for k in self.ks)
            + f" · **MRR {self.mrr:.3f}**",
            "",
        ]
        grouped = self.failures_by_stage()
        if grouped:
            lines.append("## Failure diagnosis (by stage)")
            lines.append("")
            for stage, rows in sorted(grouped.items()):
                lines.append(f"### Lost at `{stage}` — {len(rows)} query(ies)")
                for r in rows:
                    lines.append(
                        f"- \"{r.query}\" → gold {sorted(r.gold)}, "
                        f"final Top-5 {r.trace.final[:5]}"
                    )
                lines.append("")
        else:
            lines.append("No failing queries 🎉")
        return "\n".join(lines)


class EvalHarness:
    """검색 파이프라인 평가 실행기.

    run_fn: 질의 문자열을 받아 StageTrace를 반환하는 함수.
            엔진이 무엇이든(BM25, 하이브리드, LLM 재랭킹) 이 서명만 맞추면 된다.
    ks:     Hit@K를 계산할 K 값들. 마지막 단계 후보가 max(ks)보다 짧아도 동작한다.
    """

    def __init__(self, run_fn: Callable[[str], StageTrace], ks: tuple[int, ...] = (1, 5, 10)):
        if not ks:
            raise ValueError("ks must not be empty")
        self.run_fn = run_fn
        self.ks = tuple(sorted(ks))

    def evaluate(self, labelset: list[LabeledQuery]) -> EvalReport:
        results = [self._evaluate_one(labeled) for labeled in labelset]
        return EvalReport(results=results, ks=self.ks)

    def _evaluate_one(self, labeled: LabeledQuery) -> QueryResult:
        trace = self.run_fn(labeled.query)
        final = trace.final
        hits = {k: hit_at_k(final, set(labeled.gold), k) for k in self.ks}
        rr = reciprocal_rank(final, set(labeled.gold))
        lost_at = self._diagnose(trace, labeled.gold) if not hits[max(self.ks)] else None
        return QueryResult(
            query=labeled.query,
            gold=labeled.gold,
            trace=trace,
            hits=hits,
            rr=rr,
            lost_at=lost_at,
            note=labeled.note,
        )

    def _diagnose(self, trace: StageTrace, gold: frozenset[str]) -> str:
        """정답이 처음 사라진 단계를 찾는다 (깔때기 가정)."""
        for stage_name, candidates in trace.stages:
            if not gold & set(candidates):
                return stage_name
        # 모든 단계에 정답이 있는데 Hit@max(ks)에 실패했다면
        # 최종 랭킹에서 K 밖으로 밀린 것이다.
        return f"rank>{max(self.ks)}"
