"""Verify the web demo's JS engine matches the Python implementation.

    python scripts/check_js_parity.py        (requires node)

Runs the same datasets through both implementations and compares:
  - final rankings for all labelset queries
  - Hit@K / MRR metrics
  - citation verdicts and similarity scores

The web demo is only honest if it runs the real algorithms — this script
is the proof, and CI runs it on every push.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "search_demo"))

NODE_SCRIPT = """
const CG = require(process.env.CG_ROOT + "/docs/engine.js");
const D = require(process.env.CG_ROOT + "/docs/data.js");
const expected = JSON.parse(require("fs").readFileSync(process.env.CG_EXPECTED, "utf8"));
let failures = [];
const engine = CG.buildEngine(D.PRODUCTS);
const evalRes = CG.evaluate(engine, D.LABELSET);
for (const r of evalRes.results) {
  const exp = expected.rankings[r.query];
  if (JSON.stringify(exp) !== JSON.stringify(r.trace.final))
    failures.push(`ranking ${r.query}: ${JSON.stringify(exp)} != ${JSON.stringify(r.trace.final)}`);
}
for (const [k, v] of Object.entries(expected.metrics)) {
  const jsv = Math.round(evalRes.metrics[k] * 10000) / 10000;
  if (Math.abs(jsv - v) > 1e-9) failures.push(`metric ${k}: ${v} != ${jsv}`);
}
D.CITATIONS_DATA.forEach((c, i) => {
  const res = CG.checkCitation(D.CORPUS_DATA, c, 0.85);
  const exp = expected.citations[i];
  if (res.status !== exp.status || Math.abs(Math.round(res.score * 10000) / 10000 - exp.score) > 1e-9)
    failures.push(`citation ${i}: ${JSON.stringify(exp)} != ${res.status}/${res.score}`);
});
if (failures.length) { console.log(failures.join("\\n")); process.exit(1); }
console.log("JS/Python parity: ALL MATCH");
"""


def main() -> int:
    from search import load_engine  # noqa: PLC0415

    from citeguard import Citation, CitationGate, Corpus, EvalHarness, load_labelset  # noqa: PLC0415

    engine = load_engine()
    labelset = load_labelset(ROOT / "examples/search_demo/labelset.json")
    report = EvalHarness(engine.run, ks=(1, 3, 5)).evaluate(labelset)

    corpus = Corpus.from_json(ROOT / "examples/citation_demo/corpus.json")
    rows = json.loads((ROOT / "examples/citation_demo/citations.json").read_text(encoding="utf-8"))
    gate = CitationGate(corpus, fuzzy_threshold=0.85)
    gate_report = gate.run([Citation(r["source_file"], r["page"], r["quote"]) for r in rows])

    expected = {
        "rankings": {r.query: r.trace.final for r in report.results},
        "metrics": report.to_dict()["metrics"],
        "citations": [
            {"status": r.status.value, "score": round(r.score, 4)} for r in gate_report.results
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False)
        expected_path = f.name

    import os
    result = subprocess.run(
        ["node", "-e", NODE_SCRIPT],
        capture_output=True, text=True,
        env={**os.environ, "CG_ROOT": str(ROOT), "CG_EXPECTED": expected_path},
    )
    print(result.stdout.strip() or result.stderr.strip())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
