"""Golden-test regression runner.

LLM pipeline outputs drift silently whenever prompts, models, or data
change. Golden tests pin a verified output as the baseline and compare
every later run against it recursively, **catching unintended regressions
as a diff.**

Usage pattern (pytest):
    store = GoldenStore("tests/golden")
    def test_search_regression():
        actual = run_pipeline("데부꾸로 3켤레")  # "debukkuro" = work-glove slang
        assert_matches_golden(store, "debukuro", actual)

Baselines update only explicitly:
    CITEGUARD_UPDATE_GOLDEN=1 pytest        # approve an intentional change
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_UPDATE_ENV = "CITEGUARD_UPDATE_GOLDEN"


@dataclass(frozen=True)
class Difference:
    """One difference between the baseline and the actual output."""

    path: str        # e.g. "results[2].status"
    expected: object  # baseline value (None if absent)
    actual: object    # actual value (None if absent)
    kind: str         # "changed" | "missing" | "added"

    def __str__(self) -> str:
        if self.kind == "changed":
            return f"{self.path}: {self.expected!r} → {self.actual!r}"
        if self.kind == "missing":
            return f"{self.path}: in baseline but missing from output (baseline {self.expected!r})"
        return f"{self.path}: newly appeared in output ({self.actual!r})"


def diff(expected: object, actual: object, path: str = "$") -> list[Difference]:
    """Recursively compare two JSON-compatible values and list differences."""
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
    """Comparison result for one golden case."""

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
    """Stores golden baselines as JSON files.

    One case = one file (<case_id>.json), written with sort_keys so that
    git diffs stay human-readable.
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

    # ---------- comparison ----------

    def check(self, case_id: str, actual: object, update: bool | None = None) -> CaseResult:
        """Compare an actual output against the baseline.

        When update is None, the CITEGUARD_UPDATE_GOLDEN env var decides.
        If no baseline exists yet, it is saved and "new" is returned
        (first-run bootstrap).
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
    """pytest helper — raises AssertionError with a diff when the baseline differs."""
    result = store.check(case_id, actual)
    if not result.passed:
        raise AssertionError(
            f"Does not match the golden baseline.\n{result.summary()}\n"
            f"If this change is intentional, refresh the baseline with {_UPDATE_ENV}=1."
        )
