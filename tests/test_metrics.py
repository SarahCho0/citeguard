"""IR metric tests (hit@k, RR, MRR)."""

import pytest

from citeguard import hit_at_k, mrr, reciprocal_rank


class TestHitAtK:
    def test_hit_within_k(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, 3)

    def test_miss_outside_k(self):
        assert not hit_at_k(["a", "b", "c"], {"c"}, 2)

    def test_multiple_gold_any_counts(self):
        assert hit_at_k(["x", "b"], {"a", "b"}, 2)

    def test_empty_ranked(self):
        assert not hit_at_k([], {"a"}, 5)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            hit_at_k(["a"], {"a"}, 0)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_first_gold_wins(self):
        # with multiple golds, the earliest-ranked one counts
        assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(1 / 2)


class TestMRR:
    def test_mean(self):
        assert mrr([1.0, 0.5, 0.0]) == pytest.approx(0.5)

    def test_empty(self):
        assert mrr([]) == 0.0

    def test_accepts_generator(self):
        assert mrr(x for x in [1.0, 1.0]) == 1.0
