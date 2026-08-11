"""인용 검증 게이트 데모 실행 스크립트.

    python examples/citation_demo/run_gate.py

LLM이 생성했다고 가정한 리포트의 인용 6건(정상 2 + 오류 4)을
인제스트 코퍼스와 결정적으로 대조한다. LLM 호출은 0회.
게이트가 BLOCK을 반환하면 리포트는 발행되지 않는다는 시나리오.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR.parents[1]))  # citeguard 패키지 import용 (미설치 실행 대비)

from citeguard import Citation, CitationGate, Corpus  # noqa: E402


def main() -> int:
    corpus = Corpus.from_json(DEMO_DIR / "corpus.json")
    rows = json.loads((DEMO_DIR / "citations.json").read_text(encoding="utf-8"))
    citations = [
        Citation(row["source_file"], row["page"], row["quote"]) for row in rows
    ]

    gate = CitationGate(corpus, fuzzy_threshold=0.85)
    report = gate.run(citations)

    print(report.to_markdown())
    print()
    for row, result in zip(rows, report.results):
        mark = "✅" if result.passed else "⛔"
        print(f"{mark} 기대: {row['expect']:<55} 판정: {result.status.value}")

    out = DEMO_DIR / "out"
    out.mkdir(exist_ok=True)
    (out / "gate_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "gate_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"\n리포트 저장: {out / 'gate_report.json'}")

    # 오류 인용이 섞여 있으므로 게이트는 BLOCK이어야 정상
    return 0 if not report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
