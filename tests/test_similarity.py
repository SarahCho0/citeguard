"""Edit distance — metric axioms and branch-and-bound correctness.

The point of these tests: verify empirically that levenshtein satisfies
the metric axioms, and that cap-based early exit never flips a threshold
decision (result preservation).
"""

import pytest

from citeguard import levenshtein, similarity

# samples for axiom checks (Korean / English / mixed / empty strings)
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
        # d(a,a) = 0 (half of identity of indiscernibles)
        assert levenshtein(a, a) == 0

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_positivity_and_symmetry(self, a, b):
        d = levenshtein(a, b)
        assert d >= 0
        if a != b:
            assert d > 0          # d(a,b)=0 ⟺ a=b (identity of indiscernibles)
        assert d == levenshtein(b, a)  # symmetry

    @pytest.mark.parametrize("a", SAMPLES[:5])
    @pytest.mark.parametrize("b", SAMPLES[:5])
    @pytest.mark.parametrize("c", SAMPLES[:5])
    def test_triangle_inequality(self, a, b, c):
        # d(a,c) ≤ d(a,b) + d(b,c)
        assert levenshtein(a, c) <= levenshtein(a, b) + levenshtein(b, c)

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_length_lower_bound(self, a, b):
        # |len(a)-len(b)| ≤ d(a,b) — the basis for cap pruning
        assert abs(len(a) - len(b)) <= levenshtein(a, b)


class TestCapPruning:
    def test_exact_when_under_cap(self):
        assert levenshtein("kitten", "sitting", cap=5) == 3

    def test_capped_when_over(self):
        # returns cap+1 when d > cap — the "exceeds cap" verdict itself is exact
        assert levenshtein("aaaa", "zzzz", cap=2) == 3

    def test_length_gap_shortcut(self):
        assert levenshtein("ab", "abcdefgh", cap=3) == 4  # length gap 6 > cap → immediate cap+1

    @pytest.mark.parametrize("a", SAMPLES)
    @pytest.mark.parametrize("b", SAMPLES)
    def test_cap_never_flips_threshold_decision(self, a, b):
        # key property: the threshold decision (d ≤ cap?) must not depend on capping
        exact = levenshtein(a, b)
        for cap in (0, 1, 2, 5):
            capped = levenshtein(a, b, cap=cap)
            assert (capped <= cap) == (exact <= cap)
            if exact <= cap:
                assert capped == exact  # at or under the cap, the exact value is returned


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
