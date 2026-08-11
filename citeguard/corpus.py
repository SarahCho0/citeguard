"""인제스트된 원문 코퍼스.

인용 검증 게이트가 대조할 "출처의 원문"을 (파일 → 페이지 → 텍스트)
구조로 보관한다. 실제 파이프라인에서는 PDF/pptx 인제스트 결과물이
이 구조로 들어오고, 게이트는 이 코퍼스만을 근거로 인용을 판정한다.
"""

from __future__ import annotations

import json
from pathlib import Path


class Corpus:
    """파일명 → {페이지 번호 → 페이지 텍스트} 매핑."""

    def __init__(self) -> None:
        self._pages: dict[str, dict[int, str]] = {}

    # ---------- 구축 ----------

    def add_page(self, source_file: str, page: int, text: str) -> None:
        self._pages.setdefault(source_file, {})[int(page)] = text

    @classmethod
    def from_json(cls, path: str | Path) -> "Corpus":
        """{"파일명": {"1": "본문", "2": "..."}} 형태의 JSON에서 로드."""
        corpus = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for source_file, pages in data.items():
            for page, text in pages.items():
                corpus.add_page(source_file, int(page), text)
        return corpus

    @classmethod
    def from_dir(cls, path: str | Path) -> "Corpus":
        """디렉토리의 .txt 파일들을 로드. 폼피드(\\f)로 페이지를 구분한다.

        폼피드가 없으면 파일 전체를 1페이지로 취급한다.
        """
        corpus = cls()
        for txt in sorted(Path(path).glob("*.txt")):
            raw = txt.read_text(encoding="utf-8")
            for i, page_text in enumerate(raw.split("\f"), start=1):
                corpus.add_page(txt.name, i, page_text)
        return corpus

    # ---------- 조회 ----------

    @property
    def files(self) -> list[str]:
        return sorted(self._pages)

    def has_file(self, source_file: str) -> bool:
        return source_file in self._pages

    def has_page(self, source_file: str, page: int) -> bool:
        return int(page) in self._pages.get(source_file, {})

    def get_page(self, source_file: str, page: int) -> str:
        return self._pages[source_file][int(page)]

    def pages(self, source_file: str) -> dict[int, str]:
        return dict(self._pages.get(source_file, {}))

    def __len__(self) -> int:
        return sum(len(p) for p in self._pages.values())
