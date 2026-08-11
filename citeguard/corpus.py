"""The ingested source corpus.

Holds the source-of-truth text that the citation gate verifies against,
structured as (file → page → text). In a real pipeline this is the output
of PDF/pptx ingestion; the gate judges citations against this corpus and
nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path


class Corpus:
    """Mapping of file name → {page number → page text}."""

    def __init__(self) -> None:
        self._pages: dict[str, dict[int, str]] = {}

    # ---------- construction ----------

    def add_page(self, source_file: str, page: int, text: str) -> None:
        self._pages.setdefault(source_file, {})[int(page)] = text

    @classmethod
    def from_json(cls, path: str | Path) -> "Corpus":
        """Load from JSON shaped like {"file.pdf": {"1": "text", "2": "..."}}."""
        corpus = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for source_file, pages in data.items():
            for page, text in pages.items():
                corpus.add_page(source_file, int(page), text)
        return corpus

    @classmethod
    def from_dir(cls, path: str | Path) -> "Corpus":
        """Load .txt files from a directory; form-feed (\\f) separates pages.

        A file with no form feed becomes a single page 1.
        """
        corpus = cls()
        for txt in sorted(Path(path).glob("*.txt")):
            raw = txt.read_text(encoding="utf-8")
            for i, page_text in enumerate(raw.split("\f"), start=1):
                corpus.add_page(txt.name, i, page_text)
        return corpus

    # ---------- lookup ----------

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
