"""CiteGuard live demo dashboard (Streamlit).

    streamlit run app.py

Not a report viewer — this app **runs the library live**:

  🔍 Live Search    — type any query, watch every pipeline stage in real time
                      (parsing, per-channel scores, fused ranking)
  📄 Citation Gate  — edit citations in place, verdicts recompute instantly;
                      a fuzzy-threshold slider shows the decision boundary
  🏅 Golden Watch   — "modify" the engine with a fusion-weight slider and watch
                      the golden diff pinpoint ranking regressions per query
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
SEARCH_DEMO = ROOT / "examples" / "search_demo"
CITATION_DEMO = ROOT / "examples" / "citation_demo"
sys.path.insert(0, str(SEARCH_DEMO))

from search import load_engine  # noqa: E402

from citeguard import (  # noqa: E402
    Citation,
    CitationGate,
    Corpus,
    EvalHarness,
    diff,
    load_labelset,
)

st.set_page_config(page_title="CiteGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ CiteGuard — Live Demo")
st.caption("The verification layer must be deterministic · Same input → same verdict · Zero LLM calls")


# ---------------------------------------------------------------- shared loaders
@st.cache_resource
def get_products() -> dict[str, dict]:
    rows = json.loads((SEARCH_DEMO / "products.json").read_text(encoding="utf-8"))
    return {p["id"]: p for p in rows}


@st.cache_resource
def get_corpus() -> Corpus:
    return Corpus.from_json(CITATION_DEMO / "corpus.json")


def name_of(pid: str) -> str:
    p = get_products().get(pid)
    return f"{pid} · {p['name']}" if p else pid


PRODUCTS = get_products()

tab_search, tab_gate, tab_golden = st.tabs(
    ["🔍 Live Search", "📄 Citation Gate", "🏅 Golden Watch"]
)

# ================================================================ 🔍 Live Search
with tab_search:
    st.subheader("Hybrid search pipeline — opened up, live")
    st.caption(
        "Type a field order mixing slang, typos, and quantities (the demo data is Korean "
        "hardware-store jargon — e.g. 데부꾸로 = cotton work gloves). Each keystroke runs the real "
        "pipeline: quantity parsing → alias ∥ BM25 parallel retrieval (union pool) → score-fusion Top-5. "
        "Every intermediate result is shown as-is."
    )

    examples = ["데부꾸로 3켤레", "빽색 실리콘 2개", "베니다 12티 두장", "레베루 주세요", "가꾸목 다섯개"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_input("Field order", value=st.session_state.get("query", "데부꾸로 3켤레"))

    if query.strip():
        engine = load_engine()
        trace = engine.run(query)
        meta = trace.meta

        st.markdown(
            f"**Stage 0 — parsing:** `{query}` → core term `{meta['parsed']}` "
            "(quantities & noise words stripped)"
        )

        col_alias, col_bm25, col_final = st.columns(3)
        with col_alias:
            st.markdown("**Stage 1a — alias channel** (ontology exact match)")
            rows = [{"product": name_of(pid), "score": s} for pid, s in meta["alias_scored"]]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("no hits")
        with col_bm25:
            st.markdown("**Stage 1b — BM25 channel** (char-bigram statistical retrieval)")
            rows = [{"product": name_of(pid), "score": s} for pid, s in meta["bm25_scored"]]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("no hits")
        with col_final:
            st.markdown("**Stage 2 — fused Top-5** (alias 2 : BM25 1 weighting)")
            rows = [
                {"rank": i, "product": name_of(pid), "fused score": s}
                for i, (pid, s) in enumerate(meta["fused_scored"], 1)
            ]
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.warning("no candidates — lost at retrieve")

        if not trace.final:
            st.error(
                "Neither channel knows this expression → **lost at the retrieve stage**. "
                "In production, this query would be queued for ontology alias enrichment."
            )

    st.divider()
    st.markdown("**Full labelset evaluation** — run all 16 labeled queries right here")
    if st.button("▶ Run eval harness", type="primary"):
        engine = load_engine()
        labelset = load_labelset(SEARCH_DEMO / "labelset.json")
        report = EvalHarness(engine.run, ks=(1, 3, 5)).evaluate(labelset)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hit@1", f"{report.hit_rate(1):.1%}")
        m2.metric("Hit@3", f"{report.hit_rate(3):.1%}")
        m3.metric("Hit@5", f"{report.hit_rate(5):.1%}")
        m4.metric("MRR", f"{report.mrr:.3f}")

        for stage, rows in report.failures_by_stage().items():
            st.warning(
                f"**Lost at `{stage}` — {len(rows)} query(ies)**: "
                + ", ".join(f'"{r.query}"' for r in rows)
                + "  → a retrieve loss is a recall (ontology) problem, a rerank loss is a "
                "ranking problem: different fixes"
            )
        st.dataframe(
            [
                {
                    "query": r.query,
                    "gold": ", ".join(name_of(g) for g in sorted(r.gold)),
                    "top-1": name_of(r.trace.final[0]) if r.trace.final else "—",
                    "RR": r.rr,
                    "lost at": r.lost_at or "✓",
                    "note": r.note,
                }
                for r in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )

# ================================================================ 📄 Citation Gate
with tab_gate:
    st.subheader("Citation gate — edit a quote and watch the verdict flip")
    st.caption(
        "Every citation an LLM report claims (file · page · quote) is checked against the "
        "ingested corpus deterministically. Edit the table below or add rows — verdicts "
        "recompute instantly, with zero LLM calls."
    )

    corpus = get_corpus()
    with st.expander("📚 View the ingested corpus (everything the gate knows)"):
        for f in corpus.files:
            for page, text in sorted(corpus.pages(f).items()):
                st.markdown(f"**{f} — p.{page}**")
                st.text(text)

    threshold = st.slider(
        "Fuzzy threshold (similarity at or above this passes as transcription noise)",
        0.50, 1.00, 0.85, 0.01,
        help="Edit-distance similarity = 1 − d(quote, corpus window) / max(length). See MATH.md",
    )

    default_rows = [
        {"source_file": r["source_file"], "page": r["page"], "quote": r["quote"]}
        for r in json.loads((CITATION_DEMO / "citations.json").read_text(encoding="utf-8"))
    ]
    edited = st.data_editor(
        default_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "source_file": st.column_config.SelectboxColumn("source file", options=corpus.files, required=True),
            "page": st.column_config.NumberColumn("page", min_value=1, step=1),
            "quote": st.column_config.TextColumn("quote", width="large"),
        },
    )

    citations = [
        Citation(row["source_file"], int(row["page"] or 1), row["quote"] or "")
        for row in edited
        if row.get("source_file") and row.get("quote")
    ]
    if citations:
        gate = CitationGate(corpus, fuzzy_threshold=threshold)
        report = gate.run(citations)

        c1, c2, c3 = st.columns(3)
        c1.metric("Citations", report.total)
        c2.metric("Pass rate", f"{report.pass_rate:.0%}")
        c3.metric("Gate verdict", "PASS ✅" if report.ok() else "BLOCK ⛔")

        STATUS_LABEL = {
            "verified": "✅ verified — exact match in source",
            "fuzzy_match": "🟡 fuzzy — transcription-level similarity",
            "wrong_page": "📄 wrong_page — quote is real, page label is wrong",
            "quote_not_found": "⛔ not found — suspected hallucination",
            "page_not_found": "⛔ page does not exist",
            "file_not_found": "⛔ file not in corpus",
        }
        st.dataframe(
            [
                {
                    "source": r.citation.source_file,
                    "p.": str(r.citation.page)
                    if r.found_page is None
                    else f"{r.citation.page} (actually {r.found_page})",
                    "verdict": STATUS_LABEL[r.status.value],
                    "similarity": round(r.score, 2),
                    "quote": r.citation.quote,
                }
                for r in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )
        if not report.ok():
            st.error(
                f"{len(report.failures)} citation(s) failed verification — this report is "
                "blocked from publishing. In production, this gate sits in front of the "
                "report renderer."
            )

# ================================================================ 🏅 Golden Watch
with tab_golden:
    st.subheader("Golden regression watch — 'modify' the engine, catch what silently changes")
    st.caption(
        "Move the fusion weight (how much the alias channel is trusted). Metrics can stay "
        "flat while rankings shift — and the golden diff pinpoints exactly which query's "
        "which rank changed. This is why regression tests must not watch metrics alone."
    )

    weight = st.slider("Alias channel weight (baseline: 2.0)", 0.0, 4.0, 2.0, 0.25)

    engine = load_engine()
    engine.ALIAS_WEIGHT = weight
    labelset = load_labelset(SEARCH_DEMO / "labelset.json")
    report = EvalHarness(engine.run, ks=(1, 3, 5)).evaluate(labelset)

    current = {
        "metrics": report.to_dict()["metrics"],
        "final_rankings": {r.query: r.trace.final for r in report.results},
    }

    baseline_path = SEARCH_DEMO / "golden" / "search_eval_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    differences = diff(baseline, current)

    m1, m2, m3 = st.columns(3)
    m1.metric("Hit@1", f"{report.hit_rate(1):.1%}")
    m2.metric("MRR", f"{report.mrr:.3f}")
    m3.metric("Golden verdict", "PASS ✅" if not differences else f"FAIL ⛔ ({len(differences)} diffs)")

    if not differences:
        st.success("Identical to baseline — not a single ranking regressed.")
    else:
        st.error(f"{len(differences)} difference(s) vs baseline — which query, which rank:")
        st.dataframe(
            [
                {"path": d.path, "baseline": str(d.expected), "current": str(d.actual), "kind": d.kind}
                for d in differences
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "If this change is intentional, approve a new baseline with "
            "`CITEGUARD_UPDATE_GOLDEN=1` — updates are always explicit, silent drift always fails."
        )
