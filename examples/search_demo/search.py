"""데모 검색 엔진 — 현장용어 하이브리드 상품 검색.

실서비스에서 검증한 3단 구조를 공개 데이터로 축소 재현한 것:

  Stage 0. 질의 파싱      — 수량·잡음 단어 제거 (LLM 파싱의 결정적 축소판)
  Stage 1. 병렬 2채널 검색 — 별칭(alias) 정확 매칭 ∥ char n-gram BM25
                            → 두 채널의 합집합이 후보 풀 (한 채널의 실패가
                              리콜을 잃게 하지 않는 union 구조)
  Stage 2. 재랭킹          — 채널 점수 가중 융합으로 Top-5 확정
                            (실서비스에서는 이 자리가 LLM 재랭킹이며,
                             후보 풀 밖 선택을 금지하는 할루시네이션 가드를 둔다)

EvalHarness에는 깔때기 단계인 retrieve → rerank 두 단계를 노출하고,
병렬 채널별 후보는 StageTrace.meta에 기록해 디버깅에 쓴다.
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
# Stage 0 — 질의 파싱: 수량·단위·잡음 제거
# ---------------------------------------------------------------------------

# "3켤레", "100개", "두장", "열개", "12티" 중 수량 표현만 제거한다.
_QTY_PATTERN = re.compile(
    r"(\d+\s*(개|켤레|장|롤|매|입|묶음|박스|세트|자루|병|통|권|ea)|"
    r"(하나|둘|셋|다섯|열|스무)\s*(개|장|켤레)?|"
    r"(한|두|세|네|다섯|열)\s*(개|장|켤레|묶음|박스))",
    re.IGNORECASE,
)
_NOISE_WORDS = ("주세요", "주문", "대여", "부탁", "하나만", "젤", "제일", "싼", "좀")


def parse_query(query: str) -> str:
    """수량 표현과 잡음 단어를 걷어내고 상품 지칭 핵심어만 남긴다."""
    core = _QTY_PATTERN.sub(" ", query)
    for word in _NOISE_WORDS:
        core = core.replace(word, " ")
    return " ".join(core.split())


# ---------------------------------------------------------------------------
# Stage 1a — 별칭(alias) 채널: 온톨로지 기반 정확 매칭
# ---------------------------------------------------------------------------

class AliasIndex:
    """상품명·별칭을 공백 제거 정규형으로 색인해 부분 문자열 매칭한다.

    현장 은어("데부꾸로")는 형태소·오타 변형이 적고 지시 대상이 명확하므로
    통계 검색보다 사전 기반 정확 매칭이 정밀도가 높다 — 별칭 채널을
    BM25와 분리해 병렬로 두는 이유.
    """

    def __init__(self, products: list[dict]) -> None:
        # 정규형 별칭 → 상품 ID 목록
        self._index: dict[str, list[str]] = defaultdict(list)
        for product in products:
            terms = [product["name"], *product.get("aliases", [])]
            for term in terms:
                self._index[squash(term)].append(product["id"])

    def search(self, query: str) -> list[tuple[str, float]]:
        """질의 안에 색인된 별칭이 포함되면 히트. 긴 별칭일수록 높은 점수."""
        q = squash(query)
        scores: dict[str, float] = defaultdict(float)
        for term, product_ids in self._index.items():
            if term and term in q:
                for pid in product_ids:
                    # 긴 별칭 매칭 = 더 구체적인 지칭 → 질의 대비 커버리지로 점수화
                    scores[pid] = max(scores[pid], len(term) / max(len(q), 1))
        return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Stage 1b — BM25 채널: char n-gram 통계 검색
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 2) -> list[str]:
    """공백 제거 정규형의 char n-gram. 한국어 오타·붙여쓰기에 강건하다."""
    s = squash(text)
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


class BM25:
    """표준 BM25 (k1=1.5, b=0.75) — char bigram 토큰 기준.

    오타("빽색")나 사전에 없는 표기도 bigram 겹침으로 후보를 회수한다.
    별칭 채널이 놓치는 변형을 이 채널이 받쳐주는 상호 보완 구조.
    """

    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = {doc_id: Counter(char_ngrams(text)) for doc_id, text in docs.items()}
        self.doc_len = {doc_id: sum(c.values()) for doc_id, c in self.doc_tokens.items()}
        self.avg_len = sum(self.doc_len.values()) / max(len(docs), 1)
        # 문서 빈도(df) → idf
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
# 파이프라인 조립
# ---------------------------------------------------------------------------

class HybridSearch:
    """alias ∥ BM25 병렬 검색 → 점수 융합 재랭킹."""

    ALIAS_WEIGHT = 2.0  # 정확 매칭 채널을 통계 채널보다 신뢰

    def __init__(self, products: list[dict]) -> None:
        self.alias_index = AliasIndex(products)
        docs = {
            p["id"]: " ".join([p["name"], p.get("spec", ""), *p.get("aliases", [])])
            for p in products
        }
        self.bm25 = BM25(docs)

    def run(self, query: str, top_k: int = 5) -> StageTrace:
        core = parse_query(query)

        # Stage 1 — 병렬 2채널
        alias_hits = self.alias_index.search(core)
        bm25_hits = self.bm25.search(core, top_k=20)

        # union 후보 풀: 어느 한 채널만 찾아도 후보에 남는다
        pool = list(dict.fromkeys(  # 순서 보존 dedup
            [pid for pid, _ in alias_hits] + [pid for pid, _ in bm25_hits]
        ))

        # Stage 2 — 점수 융합 재랭킹 (채널별 max 정규화 후 가중합)
        fused = self._fuse(alias_hits, bm25_hits)
        reranked = [pid for pid, _ in sorted(fused.items(), key=lambda x: -x[1])][:top_k]

        return StageTrace(
            stages=[("retrieve", pool), ("rerank", reranked)],
            meta={
                "parsed": core,
                "alias_channel": [pid for pid, _ in alias_hits],
                "bm25_channel": [pid for pid, _ in bm25_hits[:10]],
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
