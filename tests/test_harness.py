"""EvalHarness — 지표 집계와 단계별 오류 진단 테스트."""

import json

import pytest

from citeguard import EvalHarness, LabeledQuery, StageTrace, load_labelset


def make_pipeline(routing: dict[str, StageTrace]):
    """질의 → 미리 정의된 궤적을 돌려주는 가짜 파이프라인."""
    def run(query: str) -> StageTrace:
        return routing[query]
    return run


class TestStageTrace:
    def test_final_is_last_stage(self):
        trace = StageTrace(stages=[("retrieve", ["a", "b"]), ("rerank", ["b"])])
        assert trace.final == ["b"]

    def test_empty_trace_final(self):
        assert StageTrace(stages=[]).final == []


class TestMetricsAggregation:
    def test_hit_rates_and_mrr(self):
        routing = {
            "q1": StageTrace([("retrieve", ["g1", "x"]), ("rerank", ["g1"])]),   # rank 1
            "q2": StageTrace([("retrieve", ["x", "g2"]), ("rerank", ["x", "g2"])]),  # rank 2
            "q3": StageTrace([("retrieve", ["x"]), ("rerank", ["x"])]),          # miss
        }
        labelset = [
            LabeledQuery("q1", frozenset({"g1"})),
            LabeledQuery("q2", frozenset({"g2"})),
            LabeledQuery("q3", frozenset({"g3"})),
        ]
        report = EvalHarness(make_pipeline(routing), ks=(1, 5)).evaluate(labelset)

        assert report.hit_rate(1) == pytest.approx(1 / 3)
        assert report.hit_rate(5) == pytest.approx(2 / 3)
        assert report.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert report.total == 3


class TestStageDiagnosis:
    def test_lost_at_retrieve(self):
        # 정답이 검색 단계부터 아예 없음 → 리콜 문제
        routing = {"q": StageTrace([("retrieve", ["x", "y"]), ("rerank", ["x"])])}
        report = EvalHarness(make_pipeline(routing), ks=(1,)).evaluate(
            [LabeledQuery("q", frozenset({"gold"}))]
        )
        assert report.results[0].lost_at == "retrieve"

    def test_lost_at_rerank(self):
        # 검색은 성공했는데 재랭킹에서 탈락 → 랭킹 문제
        routing = {"q": StageTrace([("retrieve", ["gold", "x"]), ("rerank", ["x"])])}
        report = EvalHarness(make_pipeline(routing), ks=(1,)).evaluate(
            [LabeledQuery("q", frozenset({"gold"}))]
        )
        assert report.results[0].lost_at == "rerank"

    def test_lost_beyond_k(self):
        # 모든 단계에 정답이 있으나 최종 순위가 K 밖 → rank>K로 진단
        routing = {"q": StageTrace([("retrieve", ["x", "y", "gold"]), ("rerank", ["x", "y", "gold"])])}
        report = EvalHarness(make_pipeline(routing), ks=(1, 2)).evaluate(
            [LabeledQuery("q", frozenset({"gold"}))]
        )
        assert report.results[0].lost_at == "rank>2"

    def test_success_has_no_diagnosis(self):
        routing = {"q": StageTrace([("retrieve", ["gold"]), ("rerank", ["gold"])])}
        report = EvalHarness(make_pipeline(routing), ks=(1,)).evaluate(
            [LabeledQuery("q", frozenset({"gold"}))]
        )
        assert report.results[0].lost_at is None
        assert report.results[0].passed

    def test_failures_grouped_by_stage(self):
        routing = {
            "q1": StageTrace([("retrieve", []), ("rerank", [])]),
            "q2": StageTrace([("retrieve", ["g2"]), ("rerank", ["x"])]),
            "q3": StageTrace([("retrieve", []), ("rerank", [])]),
        }
        labelset = [
            LabeledQuery("q1", frozenset({"g1"})),
            LabeledQuery("q2", frozenset({"g2"})),
            LabeledQuery("q3", frozenset({"g3"})),
        ]
        report = EvalHarness(make_pipeline(routing), ks=(1,)).evaluate(labelset)
        grouped = report.failures_by_stage()
        assert len(grouped["retrieve"]) == 2
        assert len(grouped["rerank"]) == 1


class TestReportOutput:
    def test_to_dict_shape(self):
        routing = {"q": StageTrace([("retrieve", ["g"]), ("rerank", ["g"])])}
        report = EvalHarness(make_pipeline(routing), ks=(1, 5)).evaluate(
            [LabeledQuery("q", frozenset({"g"}))]
        )
        d = report.to_dict()
        assert d["metrics"]["hit@1"] == 1.0
        assert d["metrics"]["mrr"] == 1.0
        assert d["results"][0]["lost_at"] is None

    def test_markdown_lists_failures(self):
        routing = {"q": StageTrace([("retrieve", []), ("rerank", [])])}
        report = EvalHarness(make_pipeline(routing), ks=(1,)).evaluate(
            [LabeledQuery("q", frozenset({"g"}))]
        )
        md = report.to_markdown()
        assert "retrieve" in md and "q" in md


class TestLabelset:
    def test_load_labelset(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(
            json.dumps([{"query": "니빠", "gold": ["P003"], "note": "은어"}]),
            encoding="utf-8",
        )
        labelset = load_labelset(path)
        assert labelset[0].query == "니빠"
        assert labelset[0].gold == frozenset({"P003"})

    def test_load_rejects_missing_gold(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps([{"query": "니빠"}]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_labelset(path)

    def test_empty_ks_rejected(self):
        with pytest.raises(ValueError):
            EvalHarness(lambda q: StageTrace([]), ks=())
