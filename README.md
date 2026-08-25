<div align="center">

<br/>

<img src="https://img.shields.io/badge/🛡-CiteGuard-1E3A5F?style=for-the-badge&labelColor=1C1917&color=1E3A5F" height="36"/>

<br/><br/>

# CiteGuard
### Verification Toolkit for LLM Applications

**Deterministic citation gates · Retrieval eval harness · Golden-test regression runner** —
for pipelines where *“the LLM said so”* is not evidence.

<br/>

### ▶ [**Live interactive demo**](https://sarahcho0.github.io/citeguard/) — runs the real algorithms in your browser, no install

<br/>

[![CI](https://github.com/SarahCho0/citeguard/actions/workflows/ci.yml/badge.svg)](https://github.com/SarahCho0/citeguard/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-430_passing-2E7D32?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![LLM calls](https://img.shields.io/badge/LLM_calls_in_verification-0-1E3A5F?style=flat-square)](MATH.md)
[![Dependencies](https://img.shields.io/badge/core_dependencies-0-1E3A5F?style=flat-square)](pyproject.toml)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-333333?style=flat-square)](LICENSE)

<br/>

---

</div>

<br/>

## ✦ The problem

Three failure modes show up in almost every serious LLM application:

1. **Source hallucination** — a report-writing LLM cites files, pages, and quotes… that don't exist.
2. **Unmeasured retrieval** — "the search seems to work" is not a metric, and when it fails nobody knows *which stage* failed.
3. **Silent drift** — a prompt tweak or model bump quietly changes outputs that used to be correct.

The common (and wrong) fix is to ask another LLM to check. **If you verify an LLM with another
LLM, the verifier itself is probabilistic** — the same report can pass today and fail tomorrow,
and nobody can audit why.

CiteGuard takes the opposite stance. Every verdict is a **deterministic function** of
(output, corpus), built on mathematical structure — equivalence relations, metric spaces,
tree equality — documented in [MATH.md](MATH.md) and pinned by property tests:

<table>
<tr>
  <td>🔒&nbsp;<b>Deterministic</b></td>
  <td>Same input, same verdict, every time. Zero LLM calls in the verification layer.</td>
</tr>
<tr>
  <td>🔍&nbsp;<b>Auditable</b></td>
  <td>Failures come with evidence — the matched page, the edit distance, the exact diff path.</td>
</tr>
<tr>
  <td>📐&nbsp;<b>Provable</b></td>
  <td>Performance optimizations (branch-and-bound window search) are backed by metric axioms, so they cannot change results.</td>
</tr>
</table>

This library generalizes verification patterns proven on two production AI systems at a Korean
conglomerate — a product-matching engine deployed to hardware stores, and an
investment-deliberation LLM engine used by an investment committee. All demo data here is
synthetic; no proprietary code or data is included.

---

## 🧩 The three modules

### 1 · Citation Gate — block source hallucination

Verifies every citation an LLM report claims (source file · page · quote) against the ingested
corpus by deterministic string comparison. Six verdicts: `verified` · `fuzzy_match`
(edit-distance similarity above threshold — transcription noise) · `wrong_page` (quote is real,
page label is wrong) · `quote_not_found` (suspected hallucination) · `page_not_found` ·
`file_not_found`.

```python
from citeguard import Citation, CitationGate, Corpus

corpus = Corpus.from_json("corpus.json")   # {"file.pdf": {"1": "page text", ...}}
gate = CitationGate(corpus, fuzzy_threshold=0.85)

report = gate.run([
    Citation("2025_investment_review_memo.pdf", 1,
             "Consolidated revenue for FY2024 was KRW 124.0 billion, up 18% year over year"),
])
report.ok()          # every citation must pass, or the report is not published
print(report.to_markdown())
```

### 2 · Eval Harness — Hit@K / MRR + stage-level diagnosis

Runs a labeled query set through your retrieval pipeline and reports not just *how much* fails,
but **at which stage** the gold answer was lost — a recall problem and a ranking problem need
different fixes.

```python
from citeguard import EvalHarness, load_labelset

harness = EvalHarness(run_fn=engine.run, ks=(1, 5, 10))   # run_fn: query -> StageTrace
report = harness.evaluate(load_labelset("labelset.json"))

report.hit_rate(5)            # 0.94
report.failures_by_stage()    # {"retrieve": [...], "rerank": [...]}
```

Your engine only returns a `StageTrace` (ordered stage → candidate list), so the harness is
engine-agnostic — BM25, hybrid, LLM re-ranking, anything.

### 3 · Golden Runner — catch silent regressions

Pin verified outputs as golden baselines; any later deviation fails with a readable structural
diff (`$.final_rankings["..."][2]: "P009" → "P023"`). Baselines update only on explicit
approval: `CITEGUARD_UPDATE_GOLDEN=1 pytest`.

```python
from citeguard import GoldenStore, assert_matches_golden

store = GoldenStore("tests/golden")

def test_search_regression():
    actual = engine.run("데부꾸로 3켤레").final   # "debukkuro" — see glossary below
    assert_matches_golden(store, "debukuro", actual)
```

---

## 🚀 Demos

**[Interactive web demo](https://sarahcho0.github.io/citeguard/)** — the flagship. A JavaScript
port of the engine runs client-side (parity-checked against Python in CI): edit citations and
watch verdicts flip with character-level diffs, upload your own document and verify quotes
against it, drive the search pipeline stage by stage, and move a fusion weight to watch the
golden diff catch ranking regressions in real time.

Local demos:

```bash
python examples/citation_demo/run_gate.py   # 6 citations → verdict table → BLOCK
python examples/search_demo/run_eval.py     # hybrid search eval → Hit@K + stage diagnosis
streamlit run app.py                        # Python-native dashboard
```

### About the demo data — why the queries are Korean

The search demo reproduces the production problem this library grew out of: **Korean
hardware-store workers order in Japanese-derived slang, abbreviations, and typos** that share
zero characters with official product names. A small glossary:

| Field slang | Romanized | Means | Origin |
|---|---|---|---|
| 데부꾸로 | *debukkuro* | cotton work gloves | Japanese 手袋 *tebukuro* |
| 다루끼 | *daruki* | lumber square 30×30mm | Japanese 垂木 *taruki* |
| 레베루 | *leberu* | spirit level | Japanese loan of "level" |
| 빽색 | *ppaeksaek* | white (typo of 백색) | typo |
| 가꾸목 | *kkakumok* | lumber square — **intentionally unregistered**, demonstrates retrieve-stage failure diagnosis | Japanese 角木 |

The pipeline: quantity parsing → **parallel alias-thesaurus ∥ char-bigram BM25 retrieval (union
pool)** → score-fusion re-ranking, evaluated on a 16-query labeled set. Everything else in the
repo — code, comments, reports, UI — is English; Korean appears only where Korean-text handling
*is the feature being demonstrated*.

---

## 📐 Math-grounded design

The verification layer's guarantees derive from mathematical structure, not heuristics —
documented concept-by-concept in [MATH.md](MATH.md):

- **Normalization = canonical forms of an equivalence relation** (idempotency pinned by tests)
- **Edit distance is a true metric** — symmetry, triangle inequality, and length bounds verified
  by parametrized property tests; the two-pass coarse-to-fine window search **provably returns
  the exact minimum** thanks to those axioms
- **BM25 = probability ranking principle** — RSJ odds-ratio IDF, bounded concave TF saturation,
  length-bias correction
- **Hit@K / MRR = indicator expectations**; **golden diff = ordered labeled-tree equality**
  (structural isomorphism)

---

## 🛠 Install & test

```bash
pip install -e ".[dev,demo]"    # dev = pytest · demo = streamlit (optional)
pytest                          # 430 tests, incl. metric-axiom property tests
python scripts/check_js_parity.py   # web demo JS engine ≡ Python (needs node)
```

Requires Python ≥ 3.10. The core library has **no third-party dependencies**.

## 📄 License

MIT

<br/>

---

<div align="center">

**Same input → same verdict. Every time.**
That one property is the whole toolkit.

<br/>

Verify with structure, not with another LLM 🛡

</div>
