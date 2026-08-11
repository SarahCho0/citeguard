"""Citation verification gate.

Checks every source claim in an LLM-generated report (file name · page ·
quote) against the ingested corpus, blocking source hallucination.

Core principle: **the verdict never calls an LLM.**
Every judgment is a deterministic comparison of normalized strings, so the
same input always yields the same verdict — and the result is auditable.

Verdict order:
  1. Does the file exist in the corpus?      → no: FILE_NOT_FOUND
  2. Does the page exist?                    → no: (search other pages) PAGE_NOT_FOUND
  3. Quote appears verbatim on that page    → VERIFIED
  4. Similar above the fuzzy threshold      → FUZZY_MATCH (transcription noise)
  5. Verbatim on a different page           → WRONG_PAGE (mislabeled page)
  6. Nowhere in the file                    → QUOTE_NOT_FOUND (suspected hallucination)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .corpus import Corpus
from .normalize import normalize
from .similarity import levenshtein


class Status(str, Enum):
    VERIFIED = "verified"                # exact match in the source
    FUZZY_MATCH = "fuzzy_match"          # similar above threshold (transcription noise)
    WRONG_PAGE = "wrong_page"            # quote is real but the page label is wrong
    QUOTE_NOT_FOUND = "quote_not_found"  # not in the source — suspected hallucination
    PAGE_NOT_FOUND = "page_not_found"    # the cited page does not exist
    FILE_NOT_FOUND = "file_not_found"    # the cited file is not in the corpus


# Statuses that count as passing (WRONG_PAGE is a warning, not a pass)
_PASSING = {Status.VERIFIED, Status.FUZZY_MATCH}


@dataclass(frozen=True)
class Citation:
    """One source claim made by an LLM report."""

    source_file: str
    page: int
    quote: str


@dataclass
class CitationResult:
    """Verdict for a single citation."""

    citation: Citation
    status: Status
    score: float                 # 1.0 = exact; fuzzy similarity otherwise; best ratio on failure
    matched_text: str | None = None  # the source fragment that matched (canonical form)
    found_page: int | None = None    # actual page when status is WRONG_PAGE

    @property
    def passed(self) -> bool:
        return self.status in _PASSING


@dataclass
class GateReport:
    """Gate result over all citations in a report."""

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
        """True only if every citation passes. Gate report publishing on this."""
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
    """Deterministic citation judge grounded in a corpus.

    fuzzy_threshold: similarity at or above this passes as FUZZY_MATCH.
        The default 0.85 tolerates particle/whitespace-level transcription
        noise while rejecting sentences whose content differs.
    """

    def __init__(self, corpus: Corpus, fuzzy_threshold: float = 0.85) -> None:
        if not 0.0 < fuzzy_threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be in (0, 1]")
        self.corpus = corpus
        self.fuzzy_threshold = fuzzy_threshold

    # ---------- public API ----------

    def check(self, citation: Citation) -> CitationResult:
        quote = normalize(citation.quote)

        if not self.corpus.has_file(citation.source_file):
            return CitationResult(citation, Status.FILE_NOT_FOUND, 0.0)

        if not self.corpus.has_page(citation.source_file, citation.page):
            # The cited page doesn't exist, but the quote itself may be real —
            # search the whole file and rescue as WRONG_PAGE if found.
            found = self._find_exact_in_file(citation.source_file, quote)
            if found is not None:
                return CitationResult(
                    citation, Status.WRONG_PAGE, 1.0,
                    matched_text=quote, found_page=found,
                )
            return CitationResult(citation, Status.PAGE_NOT_FOUND, 0.0)

        page_text = normalize(self.corpus.get_page(citation.source_file, citation.page))

        # 1) exact match
        if quote and quote in page_text:
            return CitationResult(citation, Status.VERIFIED, 1.0, matched_text=quote)

        # 2) fuzzy match (best similarity over a sliding window)
        ratio, window = _best_window(page_text, quote)
        if ratio >= self.fuzzy_threshold:
            return CitationResult(citation, Status.FUZZY_MATCH, ratio, matched_text=window)

        # 3) exact match on another page → mislabeled page
        found = self._find_exact_in_file(citation.source_file, quote, skip_page=citation.page)
        if found is not None:
            return CitationResult(
                citation, Status.WRONG_PAGE, 1.0,
                matched_text=quote, found_page=found,
            )

        # 4) nowhere — suspected hallucination
        return CitationResult(citation, Status.QUOTE_NOT_FOUND, ratio, matched_text=window)

    def run(self, citations: list[Citation]) -> GateReport:
        return GateReport([self.check(c) for c in citations])

    # ---------- internal ----------

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
    """Find the needle-sized window of the haystack with minimum edit distance.

    Two-pass coarse-to-fine search, exact by construction:

      Pass 1 (coarse): scan with a stride of n/4 to obtain a tight upper
      bound on the optimum quickly.
      Pass 2 (fine): scan every position (stride 1), passing the best
      distance so far as a cap so each DP aborts the moment it provably
      cannot improve.

    Because edit distance is a true metric (row-minimum monotonicity,
    length lower bound — see similarity.py), the cap-based aborts are
    result-preserving: the returned minimum equals an uncapped exhaustive
    scan. The coarse pass only tightens the cap earlier; it never decides
    the answer.
    """
    n = len(needle)
    if n == 0 or not haystack:
        return 0.0, None
    last = max(1, len(haystack) - n + 1)

    best_d: int | None = None
    best_i = 0
    # Pass 1 — coarse scan for a tight initial cap
    for i in range(0, last, max(1, n // 4)):
        cap = None if best_d is None else best_d - 1
        d = levenshtein(needle, haystack[i : i + n], cap=cap)
        if cap is None or d <= cap:
            best_d, best_i = d, i
    assert best_d is not None
    # Pass 2 — exhaustive fine scan, pruned by the running best
    for i in range(last):
        if best_d == 0:
            break  # cannot improve on an exact match
        d = levenshtein(needle, haystack[i : i + n], cap=best_d - 1)
        if d < best_d:
            best_d, best_i = d, i
    best_window = haystack[best_i : best_i + n]
    ratio = 1.0 - best_d / max(n, len(best_window))
    return ratio, best_window
