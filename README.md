# CiteGuard

**Verification toolkit for LLM applications** — deterministic citation gates, retrieval eval harness, and golden-test regression runner. **Zero LLM calls in the verification layer. Zero dependencies in the core.**

> LLM 애플리케이션 검증 툴킷 — 출처 할루시네이션을 차단하는 인용 검증 게이트,
> 라벨셋 기반 검색 평가 하네스, 파이프라인 회귀를 잡는 골든 테스트 러너.
> 검증 계층은 LLM을 1회도 호출하지 않으며, 코어는 표준 라이브러리만 사용한다.

Built from patterns proven on two production AI systems (enterprise product-matching search & an investment-deliberation LLM engine), generalized into a reusable library.

## Why deterministic verification?

If you verify an LLM with another LLM, the verifier itself becomes probabilistic —
the same report can pass today and fail tomorrow, and no one can audit why.
CiteGuard takes the opposite stance:

- **Deterministic:** every verdict is a pure function of (output, corpus). Same input, same verdict, every time.
- **Auditable:** failures come with evidence — the matched page, the edit distance, the exact diff path.
- **Provable:** performance optimizations (branch-and-bound pruning) are backed by metric-space properties, so they cannot change results. See [MATH.md](MATH.md) for the mathematical foundations (equivalence relations & canonical forms, metric axioms verified by tests, PRP-derived BM25, labeled-tree equality).

## The three modules

### 1. Citation Gate — block source hallucination

Verifies every citation (source file · page · quote) in an LLM-generated report
against the ingested corpus by deterministic string comparison.

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

Six verdicts: `verified` · `fuzzy_match` (minor transcription noise, edit-distance based) ·
`wrong_page` (quote is real, page label is wrong) · `quote_not_found` (suspected hallucination) ·
`page_not_found` · `file_not_found`.

### 2. Eval Harness — Hit@K / MRR + stage-level error diagnosis

Runs a labeled query set through your retrieval pipeline and tells you not just
*how much* fails, but **at which stage** the gold answer was lost
(recall problem vs ranking problem — they need different fixes).

```python
from citeguard import EvalHarness, load_labelset

harness = EvalHarness(run_fn=engine.run, ks=(1, 5, 10))   # run_fn: query -> StageTrace
report = harness.evaluate(load_labelset("labelset.json"))

report.hit_rate(5)            # 0.94
report.failures_by_stage()    # {"retrieve": [...], "rerank": [...]}
```

Your engine only needs to return a `StageTrace` (ordered stage → candidate list),
so the harness is engine-agnostic — BM25, hybrid, LLM re-ranking, anything.

### 3. Golden Runner — catch silent regressions

LLM pipelines drift when prompts, models, or data change. Pin verified outputs
as golden baselines; any deviation fails with a readable diff.

```python
from citeguard import GoldenStore, assert_matches_golden

store = GoldenStore("tests/golden")

def test_search_regression():
    actual = engine.run("데부꾸로 3켤레").final
    assert_matches_golden(store, "debukuro", actual)
```

Baselines update only on explicit approval: `CITEGUARD_UPDATE_GOLDEN=1 pytest`.

## Demos

Two runnable end-to-end demos (Korean field-slang product search + investment-report citations):

```bash
python examples/citation_demo/run_gate.py   # 6 citations → gate verdict table, BLOCK
python examples/search_demo/run_eval.py     # hybrid alias∥BM25 search → Hit@K + stage diagnosis
streamlit run app.py                        # dashboard over both reports
```

The search demo reproduces a production pattern: noisy field orders
("데부꾸로 3켤레", "빽색 실리콘") → quantity parsing → **parallel alias ∥ char-bigram BM25
retrieval (union pool)** → score-fusion re-ranking, evaluated on a labeled set with
one intentional failure to demonstrate stage diagnosis.

## Install & test

```bash
pip install -e ".[dev]"
pytest            # 430 tests, including metric-axiom property tests
```

Requires Python ≥ 3.10. Core has no third-party dependencies.

## Design lineage

The three modules generalize verification patterns I designed on production systems:
citation-verification gates for an investment-deliberation LLM engine (blocking
source hallucination in committee reports), and the evaluation harness + golden
suite for a field-language product-matching engine deployed to live stores.
This library is a from-scratch public reimplementation on synthetic data —
no proprietary code or data included.

## License

MIT
