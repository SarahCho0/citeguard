"""골든 테스트 러너 (Golden Test Runner).

LLM 파이프라인의 출력은 프롬프트·모델·데이터가 바뀔 때마다 조용히 변한다.
골든 테스트는 "검증된 출력"을 기준선(golden)으로 저장해 두고,
이후 실행 결과를 기준선과 재귀 비교해 **의도치 않은 회귀를 diff로 잡아낸다.**

사용 패턴 (pytest):
    store = GoldenStore("tests/golden")
    def test_search_regression():
        actual = run_pipeline("데부꾸로 3켤레")
        assert_matches_golden(store, "debukuro", actual)

기준선 갱신은 명시적으로만:
    CITEGUARD_UPDATE_GOLDEN=1 pytest        # 의도한 변경을 새 기준선으로 승인
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_UPDATE_ENV = "CITEGUARD_UPDATE_GOLDEN"


@dataclass(frozen=True)
class Difference:
    """기준선과 실제 출력의 차이 1건."""

    path: str        # 예: "results[2].status"
    expected: object  # 기준선 값 (없으면 None)
    actual: object    # 실제 값 (없으면 None)
    kind: str         # "changed" | "missing" | "added"

    def __str__(self) -> str:
        if self.kind == "changed":
            return f"{self.path}: {self.expected!r} → {self.actual!r}"
        if self.kind == "missing":
            return f"{self.path}: in baseline but missing from output (baseline {self.expected!r})"
        return f"{self.path}: newly appeared in output ({self.actual!r})"


def diff(expected: object, actual: object, path: str = "$") -> list[Difference]:
    """JSON 호환 값 두 개를 재귀 비교해 차이 목록을 반환한다."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in actual:
                differences.append(Difference(child, expected[key], None, "missing"))
            elif key not in expected:
                differences.append(Difference(child, None, actual[key], "added"))
            else:
                differences.extend(diff(expected[key], actual[key], child))
        return differences

    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for i in range(max(len(expected), len(actual))):
            child = f"{path}[{i}]"
            if i >= len(actual):
                differences.append(Difference(child, expected[i], None, "missing"))
            elif i >= len(expected):
                differences.append(Difference(child, None, actual[i], "added"))
            else:
                differences.extend(diff(expected[i], actual[i], child))
        return differences

    if expected != actual:
        return [Difference(path, expected, actual, "changed")]
    return []


@dataclass
class CaseResult:
    """골든 케이스 1건의 비교 결과."""

    case_id: str
    status: str                   # "pass" | "fail" | "new" | "updated"
    differences: list[Difference]

    @property
    def passed(self) -> bool:
        return self.status in ("pass", "new", "updated")

    def summary(self) -> str:
        if self.status == "pass":
            return f"[PASS] {self.case_id}"
        if self.status == "new":
            return f"[NEW]  {self.case_id} — baseline saved for the first time"
        if self.status == "updated":
            return f"[UPD]  {self.case_id} — baseline updated"
        lines = [f"[FAIL] {self.case_id} — {len(self.differences)} difference(s)"]
        lines += [f"  · {d}" for d in self.differences[:20]]
        if len(self.differences) > 20:
            lines.append(f"  · … and {len(self.differences) - 20} more")
        return "\n".join(lines)


class GoldenStore:
    """골든 기준선을 JSON 파일로 보관하는 저장소.

    케이스 1건 = 파일 1개(<case_id>.json). sort_keys로 저장해
    git diff가 사람이 읽을 수 있는 형태가 되도록 한다.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, case_id: str) -> Path:
        if not case_id or "/" in case_id or "\\" in case_id:
            raise ValueError(f"invalid case_id: {case_id!r}")
        return self.root / f"{case_id}.json"

    def exists(self, case_id: str) -> bool:
        return self._path(case_id).exists()

    def save(self, case_id: str, data: object) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(case_id).write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self, case_id: str) -> object:
        return json.loads(self._path(case_id).read_text(encoding="utf-8"))

    def list_cases(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    # ---------- 비교 ----------

    def check(self, case_id: str, actual: object, update: bool | None = None) -> CaseResult:
        """실제 출력을 기준선과 비교한다.

        update가 None이면 CITEGUARD_UPDATE_GOLDEN 환경변수를 따른다.
        기준선이 없으면 저장하고 "new"를 반환한다 (최초 실행 부트스트랩).
        """
        if update is None:
            update = os.environ.get(_UPDATE_ENV, "") == "1"

        if not self.exists(case_id):
            self.save(case_id, actual)
            return CaseResult(case_id, "new", [])

        if update:
            self.save(case_id, actual)
            return CaseResult(case_id, "updated", [])

        differences = diff(self.load(case_id), actual)
        status = "pass" if not differences else "fail"
        return CaseResult(case_id, status, differences)


def assert_matches_golden(store: GoldenStore, case_id: str, actual: object) -> None:
    """pytest용 헬퍼 — 기준선과 다르면 diff를 담은 AssertionError를 던진다."""
    result = store.check(case_id, actual)
    if not result.passed:
        raise AssertionError(
            f"Does not match the golden baseline.\n{result.summary()}\n"
            f"If this change is intentional, refresh the baseline with {_UPDATE_ENV}=1."
        )
