"""CiteGuard — LLM 애플리케이션 검증 툴킷.

세 개의 독립 모듈로 구성된다:

1. Citation Gate  — LLM 리포트의 출처(파일·페이지·인용문)를 원문과
                    결정적 문자열 대조로 검증 (LLM 호출 0회)
2. Eval Harness   — 라벨셋 기반 Hit@K/MRR 평가 + 단계별 오류 진단
3. Golden Runner  — 파이프라인 출력의 회귀를 diff로 잡는 골든 테스트

설계 원칙: 검증 계층은 결정적이어야 한다.
LLM으로 LLM을 검증하면 검증 자체가 비결정적이 되어 감사가 불가능해진다.
"""

from .corpus import Corpus
from .gate import Citation, CitationGate, CitationResult, GateReport, Status
from .golden import CaseResult, Difference, GoldenStore, assert_matches_golden, diff
from .harness import (
    EvalHarness,
    EvalReport,
    LabeledQuery,
    QueryResult,
    StageTrace,
    load_labelset,
)
from .metrics import hit_at_k, mrr, reciprocal_rank
from .normalize import normalize, squash
from .similarity import levenshtein, similarity

__version__ = "0.1.0"

__all__ = [
    # citation gate
    "Corpus", "Citation", "CitationGate", "CitationResult", "GateReport", "Status",
    # eval harness
    "EvalHarness", "EvalReport", "LabeledQuery", "QueryResult", "StageTrace",
    "load_labelset", "hit_at_k", "mrr", "reciprocal_rank",
    # golden
    "GoldenStore", "CaseResult", "Difference", "diff", "assert_matches_golden",
    # utils
    "normalize", "squash", "levenshtein", "similarity",
]
