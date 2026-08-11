"""normalize/squash canonicalization tests."""

import pytest

from citeguard import normalize, squash


class TestNormalize:
    def test_collapses_whitespace(self):
        assert normalize("매출액은   1,240억 원") == "매출액은 1,240억 원"

    def test_lowercases(self):
        assert normalize("EBITDA는 118억") == "ebitda는 118억"

    def test_unifies_smart_quotes(self):
        assert normalize("“인용문”") == '"인용문"'
        assert normalize("‘단일’") == "'단일'"

    def test_nfkc_fullwidth_to_halfwidth(self):
        # full-width digits/letters must unify to half-width
        assert normalize("１２３ＡＢＣ") == "123abc"

    def test_strips_edges(self):
        assert normalize("  텍스트  ") == "텍스트"

    def test_removes_zero_width(self):
        assert normalize("매​출") == "매출"

    def test_idempotent(self):
        once = normalize("“ＡＢＣ  가나다”")
        assert normalize(once) == once

    def test_empty(self):
        assert normalize("") == ""


class TestSquash:
    def test_removes_all_spaces(self):
        assert squash("백색 실리콘 300ml") == "백색실리콘300ml"

    def test_normalizes_then_squashes(self):
        assert squash("  데부 꾸로 ") == "데부꾸로"
