"""Retrieval evaluation harness.

Feeds a labeled set (query → gold product/document IDs) through a search
pipeline, aggregates Hit@K / MRR, and for every failing query diagnoses
**at which stage the gold answer was lost** (stage-level error diagnosis).

The pipeline only needs to return a StageTrace of per-stage candidate
lists, so the harness is fully decoupled from the engine implementation
(engines are swappable).

Stage diagnosis logic:
  Candidates are assumed to narrow through a funnel of stages.
  The first stage where the gold answer disappears = the failing stage.
  e.g. present in retrieve but missing in rerank → a ranking problem;
       missing from retrieve onward → a recall problem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .metrics import hit_at_k, mrr, reciprocal_rank


@dataclass
class StageTrace:
    """The trajectory of one query through the pipeline.

    stages: (stage name, candidate ID list for that stage) — in pipeline
            order. The last stage is the final ranking.
    meta:   free-form debug info (e.g. parse output, per-channel hits).
            Parallel channels (alias/BM25 etc.) are not funnel stages, so
            they belong in meta, not stages.
    """

    stages: list[tuple[str, list[str]]]
    meta: dict = field(default_factory=dict)

    @property
    def final(self) -> list[str]:
        """The final ranking (candidate list of the last stage)."""
        if not self.stages:
            return []
        return self.stages[-1][1]


@dataclass(frozen=True)
class LabeledQuery:
    """One labelset row: a real-world query and its gold ID set."""

    query: str
    gold: frozenset[str]
    note: str = ""


def load_labelset(path: str | Path) -> list[LabeledQuery]:
    """Load JSON shaped like [{"query": "...", "gold": ["P001"], "note": "..."}]."""
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
    """Evaluation result for one query."""

    query: str
    gold: frozenset[str]
    trace: StageTrace
    hits: dict[int, bool]        # k → Hit@K
    rr: float                    # reciprocal rank
    lost_at: str | None          # stage where the gold answer was lost (None on success)
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.lost_at is None


@dataclass
class EvalReport:
    """Evaluation report over the whole labelset."""

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
        """Group failures by losing stage — the basis for prioritizing fixes."""
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
    """Evaluation runner for a search pipeline.

    run_fn: a function taking a query string and returning a StageTrace.
            Any engine (BM25, hybrid, LLM re-ranked) fits this signature.
    ks:     the K values for Hit@K. Works even if the final stage has
            fewer candidates than max(ks).
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
        """Find the first stage where the gold answer disappears (funnel assumption)."""
        for stage_name, candidates in trace.stages:
            if not gold & set(candidates):
                return stage_name
        # Gold is present in every stage yet Hit@max(ks) failed — it was
        # ranked outside K in the final list.
        return f"rank>{max(self.ks)}"
