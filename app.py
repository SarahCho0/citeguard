"""CiteGuard 라이브 데모 대시보드 (Streamlit).

    streamlit run app.py

저장된 리포트를 보여주는 뷰어가 아니라, **라이브러리를 실시간으로 실행**하는 데모다:

  🔍 Live Search    — 검색어를 직접 입력 → 하이브리드 파이프라인의 단계별
                      내부(파싱·채널별 후보·융합 점수)를 실시간 시각화
  📄 Citation Gate  — 인용문을 직접 수정/추가 → 게이트가 즉시 재판정,
                      퍼지 임계값 슬라이더로 판정 경계 체험
  🏅 Golden Watch   — 융합 가중치를 돌려 엔진을 "개조" → 골든 테스트가
                      랭킹 회귀를 diff로 잡아내는 것을 실시간 확인
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
st.caption("검증 계층은 결정적이어야 한다 · Same input → same verdict · LLM 호출 0회")


# ---------------------------------------------------------------- 공용 로더
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
    st.subheader("하이브리드 검색 파이프라인 — 내부를 실시간으로 열어본다")
    st.caption(
        "현장 은어·오타·수량이 섞인 주문을 입력하면: 수량 파싱 → 별칭 ∥ BM25 병렬 검색(union) "
        "→ 점수 융합 Top-5. 각 단계의 중간 산출물을 그대로 보여준다."
    )

    examples = ["데부꾸로 3켤레", "빽색 실리콘 2개", "베니다 12티 두장", "레베루 주세요", "가꾸목 다섯개"]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_input("주문 입력", value=st.session_state.get("query", "데부꾸로 3켤레"))

    if query.strip():
        engine = load_engine()
        trace = engine.run(query)
        meta = trace.meta

        st.markdown(f"**Stage 0 — 파싱:** `{query}` → 핵심어 `{meta['parsed']}` (수량·잡음 제거)")

        col_alias, col_bm25, col_final = st.columns(3)
        with col_alias:
            st.markdown("**Stage 1a — 별칭 채널** (온톨로지 정확 매칭)")
            rows = [{"상품": name_of(pid), "점수": s} for pid, s in meta["alias_scored"]]
            st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.info("매칭 없음")
        with col_bm25:
            st.markdown("**Stage 1b — BM25 채널** (char bigram 통계 검색)")
            rows = [{"상품": name_of(pid), "점수": s} for pid, s in meta["bm25_scored"]]
            st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.info("매칭 없음")
        with col_final:
            st.markdown("**Stage 2 — 융합 Top-5** (별칭 2 : BM25 1 가중)")
            rows = [
                {"순위": i, "상품": name_of(pid), "융합점수": s}
                for i, (pid, s) in enumerate(meta["fused_scored"], 1)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True) if rows else st.warning("후보 없음 — retrieve 유실")

        if not trace.final:
            st.error(
                "두 채널 모두 이 표현을 모릅니다 → **retrieve 단계 유실**. "
                "실무에서는 이 질의가 온톨로지 별칭 보강 대상으로 큐잉됩니다."
            )

    st.divider()
    st.markdown("**라벨셋 전체 평가** — 16건을 지금 이 자리에서 실행")
    if st.button("▶ 평가 하네스 실행", type="primary"):
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
                f"**`{stage}` 단계 유실 {len(rows)}건** — "
                + ", ".join(f'"{r.query}"' for r in rows)
                + "  → retrieve 유실은 리콜(온톨로지) 문제, rerank 유실은 랭킹 문제: 처방이 다르다"
            )
        st.dataframe(
            [
                {
                    "질의": r.query,
                    "정답": ", ".join(name_of(g) for g in sorted(r.gold)),
                    "Top-1": name_of(r.trace.final[0]) if r.trace.final else "—",
                    "RR": r.rr,
                    "유실 단계": r.lost_at or "✓",
                    "비고": r.note,
                }
                for r in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )

# ================================================================ 📄 Citation Gate
with tab_gate:
    st.subheader("인용 검증 게이트 — 인용문을 직접 고쳐보면 판정이 바뀐다")
    st.caption(
        "LLM 리포트가 주장하는 출처(파일·페이지·인용문)를 원문 코퍼스와 결정적으로 대조한다. "
        "아래 표를 직접 수정하거나 행을 추가해 보라 — 판정은 즉시, LLM 호출 없이 갱신된다."
    )

    corpus = get_corpus()
    with st.expander("📚 인제스트된 원문 코퍼스 보기 (게이트가 아는 전부)"):
        for f in corpus.files:
            for page, text in sorted(corpus.pages(f).items()):
                st.markdown(f"**{f} — p.{page}**")
                st.text(text)

    threshold = st.slider(
        "퍼지 임계값 (이 유사도 이상이면 전사 오차로 보고 통과)",
        0.50, 1.00, 0.85, 0.01,
        help="편집거리 기반 유사도 = 1 − d(인용문, 원문 윈도우) / max(길이). MATH.md 참조",
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
            "source_file": st.column_config.SelectboxColumn("출처 파일", options=corpus.files, required=True),
            "page": st.column_config.NumberColumn("페이지", min_value=1, step=1),
            "quote": st.column_config.TextColumn("인용문", width="large"),
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
        c1.metric("인용 수", report.total)
        c2.metric("통과율", f"{report.pass_rate:.0%}")
        c3.metric("게이트 판정", "PASS ✅" if report.ok() else "BLOCK ⛔")

        STATUS_LABEL = {
            "verified": "✅ verified — 원문에 정확히 존재",
            "fuzzy_match": "🟡 fuzzy — 전사 오차 수준으로 유사",
            "wrong_page": "📄 wrong_page — 실재하나 페이지 표기 오류",
            "quote_not_found": "⛔ not found — 할루시네이션 의심",
            "page_not_found": "⛔ 페이지 없음",
            "file_not_found": "⛔ 파일 없음",
        }
        st.dataframe(
            [
                {
                    "출처": r.citation.source_file,
                    "p.": str(r.citation.page)
                    if r.found_page is None
                    else f"{r.citation.page} (실제 {r.found_page})",
                    "판정": STATUS_LABEL[r.status.value],
                    "유사도": round(r.score, 2),
                    "인용문": r.citation.quote,
                }
                for r in report.results
            ],
            use_container_width=True,
            hide_index=True,
        )
        if not report.ok():
            st.error(
                f"검증 실패 인용 {len(report.failures)}건 — 이 리포트는 발행이 차단됩니다. "
                "실무 파이프라인에서는 이 게이트가 리포트 렌더링 앞단에 위치합니다."
            )

# ================================================================ 🏅 Golden Watch
with tab_golden:
    st.subheader("골든 회귀 감시 — 엔진을 '개조'하면 무엇이 조용히 바뀌는지 잡아낸다")
    st.caption(
        "융합 가중치(별칭 채널 신뢰도)를 바꿔 보라. 지표는 그대로여도 랭킹이 바뀔 수 있고, "
        "골든 diff는 그 변화를 질의 단위 경로로 정확히 짚어낸다 — 회귀 테스트가 지표만 봐서는 안 되는 이유."
    )

    weight = st.slider("별칭 채널 가중치 (기준선은 2.0)", 0.0, 4.0, 2.0, 0.25)

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
    m3.metric("골든 판정", "PASS ✅" if not differences else f"FAIL ⛔ ({len(differences)} diffs)")

    if not differences:
        st.success("기준선과 완전 동일 — 랭킹 하나까지 회귀 없음.")
    else:
        st.error(f"기준선 대비 차이 {len(differences)}건 — 어떤 질의의 몇 번째 순위가 바뀌었는지:")
        st.dataframe(
            [
                {"경로": d.path, "기준선": str(d.expected), "현재": str(d.actual), "유형": d.kind}
                for d in differences
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "의도한 개선이라면 `CITEGUARD_UPDATE_GOLDEN=1`로 새 기준선을 승인한다 — "
            "갱신은 항상 명시적, 조용한 드리프트는 항상 실패."
        )
