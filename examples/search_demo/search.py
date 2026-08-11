"""Demo search engine — hybrid product search over field language.

Background: in Korean hardware stores, field workers rarely use official
product names. They order in Japanese-derived slang, abbreviations, and
typos — "데부꾸로" (debukkuro, from Japanese *tebukuro*) means cotton work
gloves; "다루끼" (daruki, from *taruki*) means a 30×30mm lumber square.
An exact-match catalog search returns nothing for these. This demo is a
scaled-down public reproduction of a production pattern that solves it:

  Stage 0. Query parsing       — strip quantities & noise words
                                 (a deterministic stand-in for LLM parsing)
  Stage 1. Two parallel channels — alias exact matching ∥ char n-gram BM25,
                                 candidate pool = the UNION of both, so one
                                 channel's miss never loses recall
  Stage 2. Re-ranking           — weighted score fusion into a Top-5
                                 (in production this slot is an LLM re-ranker,
                                 constrained to the candidate pool as a
                                 hallucination guard)

The harness sees the funnel stages retrieve → rerank; per-channel hits are
recorded in StageTrace.meta for debugging.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from citeguard import StageTrace
from citeguard.normalize import squash

# ---------------------------------------------------------------------------
# Stage 0 — query parsing: strip quantities, units, and noise
# ---------------------------------------------------------------------------

# Removes quantity expressions like "3켤레" (3 pairs), "100개" (100 pcs),
# "두장" (two sheets), "열개" (ten pcs) while leaving product terms intact.
_QTY_PATTERN = re.compile(
    r"(\d+\s*(개|켤레|장|롤|매|입|묶음|박스|세트|자루|병|통|권|ea)|"
    r"(하나|둘|셋|다섯|열|스무)\s*(개|장|켤레)?|"
    r"(한|두|세|네|다섯|열)\s*(개|장|켤레|묶음|박스))",
    re.IGNORECASE,
)
# Filler words seen in real orders: "please", "rental", "the cheapest", ...
_NOISE_WORDS = ("주세요", "주문", "대여", "부탁", "하나만", "젤", "제일", "싼", "좀")


def parse_query(query: str) -> str:
    """Strip quantity expressions and noise words, keeping product terms."""
    core = _QTY_PATTERN.sub(" ", query)
    for word in _NOISE_WORDS:
        core = core.replace(word, " ")
    return " ".join(core.split())


# ---------------------------------------------------------------------------
# Stage 1a — alias channel: ontology-based exact matching
# ---------------------------------------------------------------------------

class AliasIndex:
    """Indexes product names & aliases in squashed canonical form.

    Field slang ("데부꾸로") has few morphological variants and an
    unambiguous referent, so dictionary-based exact matching beats
    statistical retrieval on precision — which is why the alias channel
    runs in parallel with BM25 instead of being merged into it.
    """

    def __init__(self, products: list[dict]) -> None:
        # canonical alias → product IDs
        self._index: dict[str, list[str]] = defaultdict(list)
        for product in products:
            terms = [product["name"], *product.get("aliases", [])]
            for term in terms:
                self._index[squash(term)].append(product["id"])

    def search(self, query: str) -> list[tuple[str, float]]:
        """Hit when an indexed alias appears inside the query; longer aliases score higher."""
        q = squash(query)
        scores: dict[str, float] = defaultdict(float)
        for term, product_ids in self._index.items():
            if term and term in q:
                for pid in product_ids:
                    # longer matched alias = more specific reference →
                    # score by coverage of the query
                    scores[pid] = max(scores[pid], len(term) / max(len(q), 1))
        return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Stage 1b — BM25 channel: char n-gram statistical retrieval
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 2) -> list[str]:
    """Char n-grams of the squashed canonical form — robust to Korean typos and spacing."""
    s = squash(text)
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


class BM25:
    """Standard Okapi BM25 (k1=1.5, b=0.75) over char bigrams.

    Recovers typos ("빽색" for "백색"/white) and unregistered spellings
    through bigram overlap — backing up whatever the alias channel misses.
    The two channels are deliberately complementary.
    """

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = {doc_id: Counter(char_ngrams(text)) for doc_id, text in docs.items()}
        self.doc_len = {doc_id: sum(c.values()) for doc_id, c in self.doc_tokens.items()}
        self.avg_len = sum(self.doc_len.values()) / max(len(docs), 1)
        # document frequency → idf
        df: Counter = Counter()
        for counter in self.doc_tokens.values():
            df.update(counter.keys())
        n_docs = len(docs)
        self.idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_tokens = char_ngrams(query)
        scores: dict[str, float] = defaultdict(float)
        for token in q_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            for doc_id, counter in self.doc_tokens.items():
                tf = counter.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len
                )
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

class HybridSearch:
    """alias ∥ BM25 parallel retrieval → score-fusion re-ranking."""

    ALIAS_WEIGHT = 2.0  # trust the exact-match channel over the statistical one

    def __init__(self, products: list[dict]) -> None:
        self.alias_index = AliasIndex(products)
        docs = {
            p["id"]: " ".join([p["name"], p.get("spec", ""), *p.get("aliases", [])])
            for p in products
        }
        self.bm25 = BM25(docs)

    def run(self, query: str, top_k: int = 5) -> StageTrace:
        core = parse_query(query)

        # Stage 1 — two parallel channels
        alias_hits = self.alias_index.search(core)
        bm25_hits = self.bm25.search(core, top_k=20)

        # union candidate pool: a hit in either channel stays a candidate
        pool = list(dict.fromkeys(  # order-preserving dedup
            [pid for pid, _ in alias_hits] + [pid for pid, _ in bm25_hits]
        ))

        # Stage 2 — score-fusion re-ranking (per-channel max normalization, weighted sum)
        fused = self._fuse(alias_hits, bm25_hits)
        reranked = [pid for pid, _ in sorted(fused.items(), key=lambda x: -x[1])][:top_k]

        return StageTrace(
            stages=[("retrieve", pool), ("rerank", reranked)],
            meta={
                "parsed": core,
                "alias_channel": [pid for pid, _ in alias_hits],
                "bm25_channel": [pid for pid, _ in bm25_hits[:10]],
                # raw per-channel scores — for the dashboard and debugging
                "alias_scored": [(pid, round(score, 3)) for pid, score in alias_hits],
                "bm25_scored": [(pid, round(score, 3)) for pid, score in bm25_hits[:10]],
                "fused_scored": sorted(
                    ((pid, round(score, 3)) for pid, score in fused.items()),
                    key=lambda x: -x[1],
                )[:top_k],
            },
        )

    def _fuse(
        self,
        alias_hits: list[tuple[str, float]],
        bm25_hits: list[tuple[str, float]],
    ) -> dict[str, float]:
        fused: dict[str, float] = defaultdict(float)
        for hits, weight in ((alias_hits, self.ALIAS_WEIGHT), (bm25_hits, 1.0)):
            if not hits:
                continue
            top_score = hits[0][1] or 1.0
            for pid, score in hits:
                fused[pid] += weight * score / top_score
        return fused


def load_engine(products_path: str | Path | None = None) -> HybridSearch:
    path = Path(products_path or Path(__file__).parent / "products.json")
    products = json.loads(path.read_text(encoding="utf-8"))
    return HybridSearch(products)
