"""GoldenStore/diff — 골든 회귀 테스트 동작 검증."""

import pytest

from citeguard import GoldenStore, assert_matches_golden, diff


class TestDiff:
    def test_identical(self):
        assert diff({"a": 1}, {"a": 1}) == []

    def test_changed_value(self):
        differences = diff({"a": 1}, {"a": 2})
        assert len(differences) == 1
        assert differences[0].path == "$.a"
        assert differences[0].kind == "changed"

    def test_missing_key(self):
        differences = diff({"a": 1, "b": 2}, {"a": 1})
        assert differences[0].path == "$.b"
        assert differences[0].kind == "missing"

    def test_added_key(self):
        differences = diff({"a": 1}, {"a": 1, "b": 2})
        assert differences[0].kind == "added"

    def test_nested_path(self):
        differences = diff({"m": {"hit@1": 0.8}}, {"m": {"hit@1": 0.6}})
        assert differences[0].path == "$.m.hit@1"

    def test_list_element_changed(self):
        differences = diff({"r": ["a", "b"]}, {"r": ["a", "c"]})
        assert differences[0].path == "$.r[1]"

    def test_list_length_mismatch(self):
        differences = diff({"r": ["a", "b"]}, {"r": ["a"]})
        assert differences[0].kind == "missing"

    def test_type_change_is_changed(self):
        differences = diff({"a": 1}, {"a": "1"})
        assert differences[0].kind == "changed"


class TestGoldenStore:
    def test_first_run_bootstraps_baseline(self, tmp_path):
        store = GoldenStore(tmp_path)
        result = store.check("case1", {"hit@1": 0.8})
        assert result.status == "new"
        assert store.exists("case1")

    def test_same_output_passes(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.check("case1", {"hit@1": 0.8})
        assert store.check("case1", {"hit@1": 0.8}).status == "pass"

    def test_regression_fails_with_diff(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.check("case1", {"hit@1": 0.8})
        result = store.check("case1", {"hit@1": 0.6})
        assert result.status == "fail"
        assert not result.passed
        assert "$.hit@1" in str(result.differences[0])

    def test_explicit_update_rewrites_baseline(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.check("case1", {"v": 1})
        result = store.check("case1", {"v": 2}, update=True)
        assert result.status == "updated"
        assert store.load("case1") == {"v": 2}

    def test_update_via_env(self, tmp_path, monkeypatch):
        store = GoldenStore(tmp_path)
        store.check("case1", {"v": 1})
        monkeypatch.setenv("CITEGUARD_UPDATE_GOLDEN", "1")
        assert store.check("case1", {"v": 2}).status == "updated"

    def test_list_cases(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.save("b_case", {})
        store.save("a_case", {})
        assert store.list_cases() == ["a_case", "b_case"]

    def test_invalid_case_id_rejected(self, tmp_path):
        store = GoldenStore(tmp_path)
        with pytest.raises(ValueError):
            store.save("../escape", {})

    def test_korean_content_saved_readably(self, tmp_path):
        # ensure_ascii=False로 저장되어 git diff에서 한국어가 읽혀야 한다
        store = GoldenStore(tmp_path)
        store.save("case", {"query": "데부꾸로"})
        assert "데부꾸로" in (tmp_path / "case.json").read_text(encoding="utf-8")


class TestAssertHelper:
    def test_passes_silently(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.check("c", {"v": 1})
        assert_matches_golden(store, "c", {"v": 1})  # 예외 없어야 함

    def test_raises_with_diff_message(self, tmp_path):
        store = GoldenStore(tmp_path)
        store.check("c", {"v": 1})
        with pytest.raises(AssertionError, match="CITEGUARD_UPDATE_GOLDEN"):
            assert_matches_golden(store, "c", {"v": 2})
