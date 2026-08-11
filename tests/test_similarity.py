"""편집거리 — 거리공간 공리와 branch-and-bound 정확성 검증.

이 테스트의 요점: levenshtein이 수학적 거리함수(metric)의 공리를
만족함을 실측으로 확인하고, cap 조기 중단이 임계값 판정을
바꾸지 않는다(정확성 보존)는 것을 무작위 표본으로 검증한다.
"""

import pytest

from citeguard import levenshtein, similarity

# 삼각부등식 검증용 표본 (한국어/영어/혼합/빈 문자열)
SAMPLES = ["", "a", "kitten", "sitting", "매출액", "매출액은 증가", "부채비율 156%", "부채비율(180%)"]


class TestKnownValues:
    def test_classic_kitten_sitting(self):
        assert levenshtein("kitten", "sitting") == 3

    def test_identical(self):
        assert levenshtein("매출액", "매출액") == 0

    def test_empty_vs_text(self):
        assert levenshtein("", "abc") == 3

    def test_single_substitution_korean(self):
        assert levenshtein("백색 실리콘", "빽색 실리콘") == 1


class TestMetricAxioms:
    @pytest.mark.parametrize("a", SAMPLES)
    def test_identity(self, a):
        # d(a,a) = 0 (비퇴화성 절반)
        assert levenshtein(a, a) == 0

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_positivity_and_symmetry(self, a, b):
        d = levenshtein(a, b)
        assert d >= 0
        if a != b:
            assert d > 0          # d(a,b)=0 ⟺ a=b (비퇴화성)
        assert d == levenshtein(b, a)  # 대칭성

    @pytest.mark.parametrize("a", SAMPLES[:5])
    @pytest.mark.parametrize("b", SAMPLES[:5])
    @pytest.mark.parametrize("c", SAMPLES[:5])
    def test_triangle_inequality(self, a, b, c):
        # d(a,c) ≤ d(a,b) + d(b,c)
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_length_lower_bound(self, a, b):
        # |len(a)-len(b)| ≤ d(a,b) — cap 프루닝의 근거
        assert abs(len(a) - len(b)) <= levenshtein(a, b)


class TestCapPruning:
    def test_exact_when_under_cap(self):
        assert levenshtein("kitten", "sitting", cap=5) == 3

    def test_capped_when_over(self):
        # d > cap이면 cap+1 반환 — "cap 초과"라는 판정 자체는 정확
        assert levenshtein("aaaa", "zzzz", cap=2) == 3

    def test_length_gap_shortcut(self):
        assert levenshtein("ab", "abcdefgh", cap=3) == 4  # 길이차 6 > cap → 즉시 cap+1

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_cap_never_flips_threshold_decision(self, a, b):
        # 핵심 성질: 임계값 판정(d ≤ cap?)은 cap 유무와 무관하게 동일해야 한다
        exact = levenshtein(a, b)
        for cap in (0, 1, 2, 5):
            capped = levenshtein(a, b, cap=cap)
            assert (capped <= cap) == (exact <= cap)
            if exact <= cap:
                assert capped == exact  # cap 이하면 정확값 그대로


class TestSimilarity:
    def test_range(self):
        for a in SAMPLES:
            for b in SAMPLES:
                assert 0.0 <= similarity(a, b) <= 1.0

    def test_identical_is_one(self):
        assert similarity("부채비율", "부채비율") == 1.0

    def test_both_empty(self):
        assert similarity("", "") == 1.0

    def test_known_ratio(self):
        # d=1, max_len=6 → 1 - 1/6
        assert similarity("백색 실리콘", "빽색 실리콘") == pytest.approx(1 - 1 / 6)
