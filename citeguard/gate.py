"""인용 검증 게이트 (Citation Gate).

LLM이 생성한 리포트의 출처 표기(파일명·페이지·인용문)를 인제스트된
원문 코퍼스와 대조해 출처 할루시네이션을 차단한다.

핵심 원칙: **판정에 LLM을 1회도 호출하지 않는다.**
모든 판정은 정규화된 문자열의 결정적(deterministic) 대조로만 이루어지므로
같은 입력에 대해 항상 같은 결과가 나오고, 감사(audit) 가능하다.

판정 순서:
  1. 파일이 코퍼스에 존재하는가          → 없으면 FILE_NOT_FOUND
  2. 페이지가 존재하는가                 → 없으면 (타 페이지 탐색 후) PAGE_NOT_FOUND
  3. 인용문이 해당 페이지에 정확히 존재  → VERIFIED
  4. 유사도 임계값 이상으로 존재         → FUZZY_MATCH (경미한 전사 오차 허용)
  5. 같은 파일의 다른 페이지에 존재      → WRONG_PAGE (페이지 표기 오류)
  6. 어디에도 없음                       → QUOTE_NOT_FOUND (할루시네이션 의심)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .corpus import Corpus
from .normalize import normalize
from .similarity import levenshtein


class Status(str, Enum):
    VERIFIED = "verified"              # 원문에 정확히 존재
    FUZZY_MATCH = "fuzzy_match"        # 임계값 이상 유사 (전사 오차 수준)
    WRONG_PAGE = "wrong_page"          # 인용문은 실재하나 페이지 표기가 틀림
    QUOTE_NOT_FOUND = "quote_not_found"  # 인용문이 원문에 없음 — 할루시네이션 의심
    PAGE_NOT_FOUND = "page_not_found"  # 표기된 페이지가 존재하지 않음
    FILE_NOT_FOUND = "file_not_found"  # 표기된 파일이 코퍼스에 없음


# 게이트 통과로 인정하는 상태 (WRONG_PAGE는 "경고"로, 통과는 아님)
_PASSING = {Status.VERIFIED, Status.FUZZY_MATCH}


@dataclass(frozen=True)
class Citation:
    """LLM 리포트가 주장하는 출처 1건."""

    source_file: str
    page: int
    quote: str


@dataclass
class CitationResult:
    """인용 1건에 대한 판정 결과."""

    citation: Citation
    status: Status
    score: float                 # 1.0 = 정확 일치, FUZZY는 유사도, 실패는 최고 유사도
    matched_text: str | None = None  # 실제로 매칭된 원문 조각 (정규형)
    found_page: int | None = None    # WRONG_PAGE일 때 실제 발견 페이지

    @property
    def passed(self) -> bool:
        return self.status in _PASSING


@dataclass
class GateReport:
    """리포트 전체 인용에 대한 게이트 실행 결과."""

    results: list[CitationResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(r.status.value for r in self.results))

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(r.passed for r in self.results) / self.total

    @property
    def failures(self) -> list[CitationResult]:
        return [r for r in self.results if not r.passed]

    def ok(self) -> bool:
        """모든 인용이 통과해야 True. 리포트 발행 여부를 이 값으로 게이트한다."""
        return all(r.passed for r in self.results)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "ok": self.ok(),
            "counts": self.counts,
            "results": [
                {
                    "source_file": r.citation.source_file,
                    "page": r.citation.page,
                    "quote": r.citation.quote,
                    "status": r.status.value,
                    "score": round(r.score, 4),
                    "matched_text": r.matched_text,
                    "found_page": r.found_page,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Citation Gate Report",
            "",
            f"- Citations: **{self.total}**",
            f"- Pass rate: **{self.pass_rate:.0%}**",
            f"- Gate verdict: **{'PASS ✅' if self.ok() else 'BLOCK ⛔'}**",
            "",
            "| # | Source | p. | Verdict | Similarity | Quote |",
            "|---|--------|----|---------|------------|-------|",
        ]
        for i, r in enumerate(self.results, 1):
            quote = r.citation.quote if len(r.citation.quote) <= 40 else r.citation.quote[:37] + "…"
            extra = f" (actually p.{r.found_page})" if r.found_page else ""
            lines.append(
                f"| {i} | {r.citation.source_file} | {r.citation.page} "
                f"| {r.status.value}{extra} | {r.score:.2f} | {quote} |"
            )
        return "\n".join(lines)


class CitationGate:
    """코퍼스를 근거로 인용을 결정적으로 판정하는 게이트.

    fuzzy_threshold: 이 유사도 이상이면 FUZZY_MATCH로 통과.
        LLM이 인용문을 옮길 때 생기는 조사·공백 수준의 오차는 허용하되,
        내용이 다른 문장은 걸러내도록 기본값 0.85.
    """

    def __init__(self, corpus: Corpus, fuzzy_threshold: float = 0.85) -> None:
        if not 0.0 < fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be in (0, 1]")
        self.corpus = corpus
        self.fuzzy_threshold = fuzzy_threshold

    # ---------- 공개 API ----------

    def check(self, citation: Citation) -> CitationResult:
        quote = normalize(citation.quote)

        if not self.corpus.has_file(citation.source_file):
            return CitationResult(citation, Status.FILE_NOT_FOUND, 0.0)

        if not self.corpus.has_page(citation.source_file, citation.page):
            # 페이지 표기가 아예 없는 경우에도, 인용문 자체는 실재할 수 있으므로
            # 파일 전체를 탐색해 WRONG_PAGE로 구제한다.
            found = self._find_exact_in_file(citation.source_file, quote)
            if found is not None:
                return CitationResult(
                    citation, Status.WRONG_PAGE, 1.0,
                    matched_text=quote, found_page=found,
                )
            return CitationResult(citation, Status.PAGE_NOT_FOUND, 0.0)

        page_text = normalize(self.corpus.get_page(citation.source_file, citation.page))

        # 1) 정확 일치
        if quote and quote in page_text:
            return CitationResult(citation, Status.VERIFIED, 1.0, matched_text=quote)

        # 2) 퍼지 일치 (슬라이딩 윈도우 최고 유사도)
        ratio, window = _best_window(page_text, quote)
        if ratio >= self.fuzzy_threshold:
            return CitationResult(citation, Status.FUZZY_MATCH, ratio, matched_text=window)

        # 3) 다른 페이지에 정확히 존재하는지 확인 → 페이지 표기 오류
        found = self._find_exact_in_file(citation.source_file, quote, skip_page=citation.page)
        if found is not None:
            return CitationResult(
                citation, Status.WRONG_PAGE, 1.0,
                matched_text=quote, found_page=found,
            )

        # 4) 어디에도 없음 — 할루시네이션 의심
        return CitationResult(citation, Status.QUOTE_NOT_FOUND, ratio, matched_text=window)

    def run(self, citations: list[Citation]) -> GateReport:
        return GateReport([self.check(c) for c in citations])

    # ---------- 내부 ----------

    def _find_exact_in_file(
        self, source_file: str, quote: str, skip_page: int | None = None
    ) -> int | None:
        if not quote:
            return None
        for page, text in sorted(self.corpus.pages(source_file).items()):
            if page == skip_page:
                continue
            if quote in normalize(text):
                return page
        return None


def _best_window(haystack: str, needle: str) -> tuple[float, str | None]:
    """haystack 위를 needle 길이의 윈도우로 훑으며 최소 편집거리 윈도우를 찾는다.

    branch-and-bound: 지금까지의 최소 거리(best_d)를 cap으로 넘겨
    그보다 나빠질 비교는 DP 도중에 중단한다. 편집거리가 거리함수라는
    사실(행 최솟값 단조성·길이 하한)이 이 가지치기의 정확성을 보장하므로,
    결과는 전 윈도우 완전 탐색과 동일하다 — similarity.py 참조.
    """
    n = len(needle)
    if n == 0 or not haystack:
        return 0.0, None
    step = max(1, n // 4)
    best_d: int | None = None
    best_window: str | None = None
    for i in range(0, max(1, len(haystack) - n + 1), step):
        window = haystack[i : i + n]
        cap = None if best_d is None else best_d - 1  # 현재 최적보다 나은 경우만 정확히 계산
        d = levenshtein(needle, window, cap=cap)
        if cap is not None and d > cap:  # cap 초과 → 개선 없음, 버림
            continue
        best_d, best_window = d, window
    assert best_d is not None and best_window is not None
    ratio = 1.0 - best_d / max(n, len(best_window))
    return ratio, best_window
