"""CitationGate verdict logic tests — all six statuses covered.

Fixture corpus/quotes are Korean on purpose: Korean-text robustness is a
stated feature, and this pins it."""

import pytest

from citeguard import Citation, CitationGate, Corpus, Status


@pytest.fixture
def corpus():
    c = Corpus()
    c.add_page(
        "report.pdf", 1,
        "2024년 연결 기준 매출액은 1,240억 원으로 전년 대비 18% 증가하였다. "
        "주요 고객은 건설사이다.",
    )
    c.add_page(
        "report.pdf", 2,
        "부채비율은 156%로 업계 평균(180%)을 하회한다. 영업이익률은 6.8%이다.",
    )
    return c


@pytest.fixture
def gate(corpus):
    return CitationGate(corpus, fuzzy_threshold=0.85)


class TestVerified:
    def test_exact_quote_passes(self, gate):
        result = gate.check(Citation("report.pdf", 1, "매출액은 1,240억 원으로 전년 대비 18% 증가하였다"))
        assert result.status == Status.VERIFIED
        assert result.score == 1.0
        assert result.passed

    def test_whitespace_variation_still_verified(self, gate):
        # whitespace-count differences are absorbed by normalization
        result = gate.check(Citation("report.pdf", 1, "매출액은  1,240억   원으로 전년 대비 18% 증가하였다"))
        assert result.status == Status.VERIFIED


class TestFuzzy:
    def test_minor_transcription_error_is_fuzzy(self, gate):
        # parentheses dropped — minor transcription noise above the threshold
        result = gate.check(Citation("report.pdf", 2, "부채비율은 156%로 업계 평균 180%를 하회한다"))
        assert result.status == Status.FUZZY_MATCH
        assert 0.85 <= result.score < 1.0
        assert result.passed

    def test_fabricated_content_not_fuzzy(self, gate):
        # a sentence with a fabricated figure must stay below the threshold
        result = gate.check(Citation("report.pdf", 2, "영업이익률은 12.4%로 업계 최고 수준이다"))
        assert result.status == Status.QUOTE_NOT_FOUND
        assert not result.passed


class TestWrongPage:
    def test_quote_on_other_page(self, gate):
        result = gate.check(Citation("report.pdf", 1, "부채비율은 156%로 업계 평균(180%)을 하회한다"))
        assert result.status == Status.WRONG_PAGE
        assert result.found_page == 2
        assert not result.passed  # a mislabeled page is a warning, not a pass

    def test_missing_page_but_quote_exists_elsewhere(self, gate):
        # cited page does not exist, but the quote is real → rescued as WRONG_PAGE
        result = gate.check(Citation("report.pdf", 9, "영업이익률은 6.8%이다"))
        assert result.status == Status.WRONG_PAGE
        assert result.found_page == 2


class TestNotFound:
    def test_file_not_found(self, gate):
        result = gate.check(Citation("ghost.pdf", 1, "아무 문장"))
        assert result.status == Status.FILE_NOT_FOUND
        assert result.score == 0.0

    def test_page_not_found(self, gate):
        result = gate.check(Citation("report.pdf", 9, "코퍼스에 없는 문장"))
        assert result.status == Status.PAGE_NOT_FOUND

    def test_quote_not_found_reports_best_ratio(self, gate):
        result = gate.check(Citation("report.pdf", 1, "완전히 무관한 지어낸 문장이다"))
        assert result.status == Status.QUOTE_NOT_FOUND
        assert 0.0 <= result.score < 0.85


class TestGateReport:
    def test_report_blocks_on_any_failure(self, gate):
        report = gate.run([
            Citation("report.pdf", 1, "매출액은 1,240억 원으로 전년 대비 18% 증가하였다"),
            Citation("ghost.pdf", 1, "지어낸 인용"),
        ])
        assert not report.ok()
        assert report.pass_rate == 0.5
        assert len(report.failures) == 1

    def test_report_passes_when_all_verified(self, gate):
        report = gate.run([
            Citation("report.pdf", 1, "주요 고객은 건설사이다"),
            Citation("report.pdf", 2, "영업이익률은 6.8%이다"),
        ])
        assert report.ok()
        assert report.pass_rate == 1.0

    def test_empty_report_is_ok(self, gate):
        assert gate.run([]).ok()

    def test_to_dict_roundtrip(self, gate):
        report = gate.run([Citation("report.pdf", 1, "주요 고객은 건설사이다")])
        d = report.to_dict()
        assert d["total"] == 1
        assert d["ok"] is True
        assert d["results"][0]["status"] == "verified"

    def test_to_markdown_contains_verdict(self, gate):
        report = gate.run([Citation("ghost.pdf", 1, "x")])
        assert "BLOCK" in report.to_markdown()


class TestConfig:
    def test_invalid_threshold_rejected(self, corpus):
        with pytest.raises(ValueError):
            CitationGate(corpus, fuzzy_threshold=0.0)
        with pytest.raises(ValueError):
            CitationGate(corpus, fuzzy_threshold=1.5)

    def test_threshold_controls_fuzzy_acceptance(self, corpus):
        quote = "부채비율은 156%로 업계 평균 180%를 하회한다"
        loose = CitationGate(corpus, fuzzy_threshold=0.7).check(Citation("report.pdf", 2, quote))
        strict = CitationGate(corpus, fuzzy_threshold=0.99).check(Citation("report.pdf", 2, quote))
        assert loose.status == Status.FUZZY_MATCH
        assert strict.status == Status.QUOTE_NOT_FOUND
