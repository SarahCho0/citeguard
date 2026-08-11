"""검색 평가 하네스 데모 실행 스크립트.

    python examples/search_demo/run_eval.py

1. 하이브리드 검색 엔진을 라벨셋 15건으로 평가 (Hit@1/3/5 + MRR)
2. 실패 질의를 단계별(retrieve/rerank)로 진단
3. 결과를 골든 기준선과 비교해 회귀 여부 확인
4. 리포트를 out/eval_report.{md,json}에 저장 (Streamlit 대시보드가 읽음)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR))          # search 모듈 import용
sys.path.insert(0, str(DEMO_DIR.parents[1]))  # citeguard 패키지 import용 (미설치 실행 대비)

from search import load_engine  # noqa: E402

from citeguard import EvalHarness, GoldenStore, load_labelset  # noqa: E402


def main() -> int:
    engine = load_engine()
    labelset = load_labelset(DEMO_DIR / "labelset.json")

    harness = EvalHarness(run_fn=engine.run, ks=(1, 3, 5))
    report = harness.evaluate(labelset)

    print(report.to_markdown())
    print()

    # ---- 골든 회귀 체크: 지표와 최종 랭킹이 조용히 퇴행하지 않는지 고정 ----
    store = GoldenStore(DEMO_DIR / "golden")
    golden_payload = {
        "metrics": report.to_dict()["metrics"],
        "final_rankings": {
            r.query: r.trace.final for r in report.results
        },
    }
    case = store.check("search_eval_baseline", golden_payload)
    print(case.summary())

    # ---- 대시보드용 산출물 저장 ----
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
