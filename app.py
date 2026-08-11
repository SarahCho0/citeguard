"""CiteGuard 데모 대시보드 (Streamlit).

    streamlit run app.py

두 데모 스크립트가 저장한 리포트(JSON)를 시각화한다.
리포트가 없으면 먼저 실행:
    python examples/citation_demo/run_gate.py
    python examples/search_demo/run_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
GATE_REPORT = ROOT / "examples" / "citation_demo" / "out" / "gate_report.json"
EVAL_REPORT = ROOT / "examples" / "search_demo" / "out" / "eval_report.json"

st.set_page_config(page_title="CiteGuard", page_icon="🛡️", layout="wide")
st.title("🛡️ CiteGuard — LLM 검증 대시보드")
st.caption("결정적 인용 검증 · 검색 평가 하네스 · LLM 호출 0회")


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


tab_gate, tab_eval = st.tabs(["📄 Citation Gate", "🔍 Retrieval Eval"])

# ---------------------------------------------------------------- Citation Gate
with tab_gate:
    data = load(GATE_REPORT)
    if data is None:
        st.info("리포트가 없습니다. 먼저 실행하세요:\n\n`python examples/citation_demo/run_gate.py`")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("인용 수", data["total"])
        col2.metric("통과율", f"{data['pass_rate']:.0%}")
        col3.metric("게이트 판정", "PASS ✅" if data["ok"] else "BLOCK ⛔")

        st.subheader("상태 분포")
        st.bar_chart(data["counts"])

        st.subheader("인용별 판정")
        rows = []
        for r in data["results"]:
            rows.append({
                "출처": r["source_file"],
                "페이지": r["page"],
                "상태": r["status"] + (f" (실제 p.{r['found_page']})" if r.get("found_page") else ""),
                "유사도": r["score"],
                "인용문": r["quote"],
            })
        st.dataframe(rows, use_container_width=True)

        blocked = [r for r in data["results"] if r["status"] not in ("verified", "fuzzy_match")]
        if blocked:
            st.error(
                f"차단 사유 {len(blocked)}건 — 이 리포트는 발행되면 안 됩니다. "
                "출처가 검증되지 않은 주장(할루시네이션 의심)이 포함되어 있습니다."
            )

# ---------------------------------------------------------------- Retrieval Eval
with tab_eval:
    data = load(EVAL_REPORT)
    if data is None:
        st.info("리포트가 없습니다. 먼저 실행하세요:\n\n`python examples/search_demo/run_eval.py`")
    else:
        metrics = data["metrics"]
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.1%}" if name.startswith("hit@") else f"{value:.3f}")

        st.subheader("단계별 실패 진단")
        failures = data["failures_by_stage"]
        if not failures:
            st.success("실패 질의 없음 🎉")
        else:
            for stage, queries in failures.items():
                st.warning(
                    f"**`{stage}` 단계에서 유실 — {len(queries)}건**: "
                    + ", ".join(f'"{q}"' for q in queries)
                )
            st.caption(
                "retrieve 유실 = 리콜 문제(온톨로지 별칭·색인 보강), "
                "rerank 유실 = 랭킹 문제(융합 가중치·재랭커 개선) — 처방이 다릅니다."
            )

        st.subheader("질의별 결과")
        rows = []
        for r in data["results"]:
            rows.append({
                "질의": r["query"],
                "정답": ", ".join(r["gold"]),
                "최종 Top-5": ", ".join(r["final_top5"]) or "(없음)",
                "Hit@1": "✅" if r["hits"].get("hit@1") else "—",
                "RR": r["rr"],
                "유실 단계": r["lost_at"] or "",
                "비고": r["note"],
            })
        st.dataframe(rows, use_container_width=True)
