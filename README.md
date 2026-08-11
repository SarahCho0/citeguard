<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=200&section=header&text=CiteGuard&fontColor=ffffff&fontSize=52&fontAlignY=38&desc=Verification%20Toolkit%20for%20LLM%20Applications&descSize=20&descAlignY=60&descColor=e0e7ff" width="100%"/>

<div align="center">

![Tests](https://img.shields.io/badge/tests-430%20passing-brightgreen?style=for-the-badge)
![LLM calls](https://img.shields.io/badge/LLM%20calls%20in%20verification-0-6366f1?style=for-the-badge)
![Dependencies](https://img.shields.io/badge/core%20dependencies-0-6366f1?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-black?style=for-the-badge)

**Deterministic citation gates · Retrieval eval harness · Golden-test regression runner**

</div>

> 🛡️ LLM 애플리케이션 검증 툴킷 — 출처 할루시네이션을 차단하는 인용 검증 게이트,
> 라벨셋 기반 검색 평가 하네스, 파이프라인 회귀를 잡는 골든 테스트 러너.
> 검증 계층은 LLM을 1회도 호출하지 않으며, 코어는 표준 라이브러리만 사용한다.

Built from patterns proven on two production AI systems (enterprise product-matching search & an investment-deliberation LLM engine), generalized into a reusable library.

---

## 💡 Why deterministic verification?

> **If you verify an LLM with another LLM, the verifier itself becomes probabilistic** —
> the same report can pass today and fail tomorrow, and no one can audit why.

CiteGuard takes the opposite stance:

<table>
<tr>
  <td>🔒&nbsp;<b>Deterministic</b></td>
  <td>Every verdict is a pure function of (output, corpus). Same input, same verdict, every time.</td>
</tr>
<tr>
  <td>🔍&nbsp;<b>Auditable</b></td>
  <td>Failures come with evidence — the matched page, the edit distance, the exact diff path.</td>
</tr>
<tr>
  <td>📐&nbsp;<b>Provable</b></td>
  <td>Performance optimizations (branch-and-bound pruning) are backed by metric-space properties, so they cannot change results. See <a href="MATH.md">MATH.md</a> for the mathematical foundations — equivalence relations &amp; canonical forms, metric axioms verified by tests, PRP-derived BM25, labeled-tree equality.</td>
</tr>
</table>

---

## 🧩 The three modules

<table>
<tr>
<td width="33%" valign="top">

### 📄 Citation Gate
**Block source hallucination**

Verifies every citation (source file · page · quote) in an LLM-generated report against the ingested corpus by deterministic string comparison.

Six verdicts: `verified` · `fuzzy_match` (edit-distance based) · `wrong_page` · `quote_not_found` (suspected hallucination) · `page_not_found` · `file_not_found`

</td>
<td width="33%" valign="top">

### 🔍 Eval Harness
**Hit@K / MRR + stage diagnosis**

Runs a labeled query set through your retrieval pipeline and tells you not just *how much* fails, but **at which stage** the gold answer was lost — recall problem vs ranking problem need different fixes.

Engine-agnostic: your pipeline only returns a `StageTrace`.

</td>
<td width="33%" valign="top">

### 🏅 Golden Runner
**Catch silent regressions**

LLM pipelines drift when prompts, models, or data change. Pin verified outputs as golden baselines; any deviation fails with a readable structural diff.

Baselines update only on explicit approval:
`CITEGUARD_UPDATE_GOLDEN=1 pytest`

</td>
</tr>
</table>

### Quick start

```python
from citeguard import Citation, CitationGate, Corpus

corpus = Corpus.from_json("corpus.json")          # {"file.pdf": {"1": "page text", ...}}
gate = CitationGate(corpus, fuzzy_threshold=0.85)

report = gate.run([
    Citation("2025_투자검토보고서.pdf", 1, "매출액은 1,240억 원으로 전년 대비 18% 증가하였다"),
])
report.ok()        # False → do NOT publish the report
print(report.to_markdown())
```

```python
from citeguard import EvalHarness, load_labelset

harness = EvalHarness(run_fn=engine.run, ks=(1, 5, 10))   # run_fn: query -> StageTrace
report = harness.evaluate(load_labelset("labelset.json"))

report.hit_rate(5)            # 0.94
report.failures_by_stage()    # {"retrieve": [...], "rerank": [...]}
```

```python
from citeguard import GoldenStore, assert_matches_golden

store = GoldenStore("tests/golden")

def test_search_regression():
    actual = engine.run("데부꾸로 3켤레").final
    assert_matches_golden(store, "debukuro", actual)
```

---

## 🚀 Demos

Two runnable end-to-end demos — Korean field-slang product search + investment-report citations:

```bash
python examples/citation_demo/run_gate.py   # 6 citations → gate verdict table, BLOCK
python examples/search_demo/run_eval.py     # hybrid alias∥BM25 search → Hit@K + stage diagnosis
streamlit run app.py                        # dashboard over both reports
```

The search demo reproduces a production pattern: noisy field orders
(**"데부꾸로 3켤레"**, **"빽색 실리콘"**) → quantity parsing → **parallel alias ∥ char-bigram BM25
retrieval (union pool)** → score-fusion re-ranking, evaluated on a labeled set with
one intentional failure to demonstrate stage diagnosis.

---

## 📐 Math-grounded design

The verification layer's guarantees derive from mathematical structure, not heuristics — documented concept-by-concept in [MATH.md](MATH.md):

- **Normalization = canonical forms of an equivalence relation** (idempotency pinned by tests)
- **Edit distance = a true metric** — axioms (symmetry, triangle inequality, length lower bound) verified by parametrized property tests; branch-and-bound window search **provably returns the exact minimum**
- **BM25 = probability ranking principle** — RSJ odds-ratio IDF, bounded concave TF saturation, length-bias correction
- **Hit@K / MRR = indicator expectations**; **golden diff = ordered labeled-tree equality** (structural isomorphism)

---

## 🛠 Install & test

```bash
pip install -e ".[dev]"
pytest            # 430 tests, including metric-axiom property tests
```

Requires Python ≥ 3.10. Core has no third-party dependencies.

---

## 🧬 Design lineage

The three modules generalize verification patterns I designed on production systems:
citation-verification gates for an investment-deliberation LLM engine (blocking
source hallucination in committee reports), and the evaluation harness + golden
suite for a field-language product-matching engine deployed to live stores.
This library is a from-scratch public reimplementation on synthetic data —
no proprietary code or data included.

## 📄 License

MIT

<img src="https://capsule-render.vercel.app/api?type=waving&color=6366f1&height=100&section=footer" width="100%"/>
