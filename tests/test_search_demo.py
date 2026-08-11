"""데모 하이브리드 검색 엔진 통합 테스트.

라이브러리(EvalHarness)와 데모 엔진(alias ∥ BM25)이 실제로 맞물려
도는지 엔드투엔드로 확인한다.
"""

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "search_demo"
sys.path.insert(0, str(DEMO_DIR))

from search import AliasIndex, BM25, char_ngrams, load_engine, parse_query  # noqa: E402

from citeguard import EvalHarness, load_labelset  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    return load_engine()


class TestParseQuery:
    @pytest.mark.parametrize("query,expected", [
        ("데부꾸로 3켤레", "데부꾸로"),
        ("타이랩 100개", "타이랩"),
        ("니빠 하나", "니빠"),
        ("뿌레카 대여", "뿌레카"),
    ])
    def test_strips_quantity_and_noise(self, query, expected):
        assert parse_query(query) == expected


class TestChannels:
    def test_char_ngrams(self):
        assert char_ngrams("목장갑") == ["목장", "장갑"]

    def test_char_ngrams_short_text(self):
        assert char_ngrams("몽") == ["몽"]

    def test_alias_exact_hit(self, engine):
        hits = engine.alias_index.search("데부꾸로")
        assert hits[0][0] == "P001"

    def test_bm25_recovers_typo(self, engine):
        # "빽색"(오타)도 bigram 겹침으로 백색 실리콘을 회수해야 한다
        ids = [pid for pid, _ in engine.bm25.search("빽색 실리콘")]
        assert "P009" in ids

    def test_union_pool_keeps_both_channels(self, engine):
        trace = engine.run("함마드릴")
        pool = dict(trace.stages)["retrieve"]
        assert set(trace.meta["alias_channel"]) <= set(pool)
        assert set(trace.meta["bm25_channel"]) <= set(pool)


class TestPipeline:
    def test_trace_has_funnel_stages(self, engine):
        trace = engine.run("안전모 5개")
        assert [name for name, _ in trace.stages] == ["retrieve", "rerank"]
        assert len(trace.final) <= 5

    @pytest.mark.parametrize("query,gold", [
        ("데부꾸로 3켤레", "P001"),
        ("함마드릴 1개", "P002"),
        ("레베루 주세요", "P016"),
        ("베니다 12티 두장", "P021"),
    ])
    def test_slang_queries_rank_top1(self, engine, query, gold):
        assert engine.run(query).final[0] == gold


class TestEndToEnd:
    def test_labelset_eval_meets_quality_bar(self, engine):
        # 데모 라벨셋 기준 최소 품질 기준선 — 이 아래로 떨어지면 회귀
        labelset = load_labelset(DEMO_DIR / "labelset.json")
        report = EvalHarness(engine.run, ks=(1, 3, 5)).evaluate(labelset)
        assert report.hit_rate(5) >= 0.9
        assert report.hit_rate(1) >= 0.7
        assert report.mrr >= 0.8
