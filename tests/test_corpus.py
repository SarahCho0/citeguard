"""Corpus 로딩·조회 테스트."""

import json

import pytest

from citeguard import Corpus


@pytest.fixture
def corpus():
    c = Corpus()
    c.add_page("report.pdf", 1, "1페이지 본문")
    c.add_page("report.pdf", 2, "2페이지 본문")
    c.add_page("summary.pdf", 1, "요약 본문")
    return c


class TestCorpus:
    def test_has_file(self, corpus):
        assert corpus.has_file("report.pdf")
        assert not corpus.has_file("ghost.pdf")

    def test_has_page(self, corpus):
        assert corpus.has_page("report.pdf", 2)
        assert not corpus.has_page("report.pdf", 3)
        assert not corpus.has_page("ghost.pdf", 1)

    def test_get_page(self, corpus):
        assert corpus.get_page("report.pdf", 1) == "1페이지 본문"

    def test_files_sorted(self, corpus):
        assert corpus.files == ["report.pdf", "summary.pdf"]

    def test_len_counts_pages(self, corpus):
        assert len(corpus) == 3

    def test_pages_returns_copy(self, corpus):
        pages = corpus.pages("report.pdf")
        pages[99] = "변조"
        assert not corpus.has_page("report.pdf", 99)

    def test_page_key_accepts_str_int(self):
        c = Corpus()
        c.add_page("a.pdf", "3", "본문")  # 문자열 페이지 번호도 int로 통일
        assert c.has_page("a.pdf", 3)


class TestCorpusLoading:
    def test_from_json(self, tmp_path):
        path = tmp_path / "corpus.json"
        path.write_text(
            json.dumps({"a.pdf": {"1": "첫 페이지", "2": "둘째 페이지"}}),
            encoding="utf-8",
        )
        corpus = Corpus.from_json(path)
        assert corpus.get_page("a.pdf", 2) == "둘째 페이지"

    def test_from_dir_formfeed_pages(self, tmp_path):
        (tmp_path / "doc.txt").write_text("1페이지\f2페이지\f3페이지", encoding="utf-8")
        corpus = Corpus.from_dir(tmp_path)
        assert len(corpus) == 3
        assert corpus.get_page("doc.txt", 3) == "3페이지"

    def test_from_dir_no_formfeed_single_page(self, tmp_path):
        (tmp_path / "doc.txt").write_text("전체가 한 페이지", encoding="utf-8")
        corpus = Corpus.from_dir(tmp_path)
        assert corpus.get_page("doc.txt", 1) == "전체가 한 페이지"
