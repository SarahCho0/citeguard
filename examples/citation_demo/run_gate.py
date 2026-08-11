"""Citation gate demo runner.

    python examples/citation_demo/run_gate.py

Verifies 6 citations from a hypothetical LLM-generated investment report
(2 valid + 4 defective) against the ingested corpus — deterministically,
with zero LLM calls. If the gate returns BLOCK, the report is not
published: that is the scenario this demo plays out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR.parents[1]))  # allow running without installing the package

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
        print(f"{mark} expected: {row['expect']:<62} verdict: {result.status.value}")

    out = DEMO_DIR / "out"
    out.mkdir(exist_ok=True)
    (out / "gate_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "gate_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"\nReport saved: {out / 'gate_report.json'}")

    # Defective citations are present, so BLOCK is the correct outcome
    return 0 if not report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
