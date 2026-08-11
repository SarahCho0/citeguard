"""Retrieval eval harness demo runner.

    python examples/search_demo/run_eval.py

1. Evaluates the hybrid search engine on a 16-query labeled set (Hit@1/3/5 + MRR)
2. Diagnoses failing queries by stage (retrieve vs rerank)
3. Checks the result against the golden baseline for regressions
4. Saves reports to out/eval_report.{md,json} (read by the Streamlit dashboard)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR))          # import the local search module
sys.path.insert(0, str(DEMO_DIR.parents[1]))  # allow running without installing the package

from search import load_engine  # noqa: E402

from citeguard import EvalHarness, GoldenStore, load_labelset  # noqa: E402


def main() -> int:
    engine = load_engine()
    labelset = load_labelset(DEMO_DIR / "labelset.json")

    harness = EvalHarness(run_fn=engine.run, ks=(1, 3, 5))
    report = harness.evaluate(labelset)

    print(report.to_markdown())
    print()

    # ---- golden regression check: pin metrics AND final rankings so nothing drifts silently
    store = GoldenStore(DEMO_DIR / "golden")
    golden_payload = {
        "metrics": report.to_dict()["metrics"],
        "final_rankings": {
            r.query: r.trace.final for r in report.results
        },
    }
    case = store.check("search_eval_baseline", golden_payload)
    print(case.summary())

    # ---- save artifacts for the dashboard
    out = DEMO_DIR / "out"
    out.mkdir(exist_ok=True)
    (out / "eval_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "eval_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nReport saved: {out / 'eval_report.json'}")

    return 0 if case.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
