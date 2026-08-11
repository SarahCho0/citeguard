"""CiteGuard — verification toolkit for LLM applications.

Three independent modules:

1. Citation Gate  — verifies the sources an LLM report claims (file ·
                    page · quote) against the ingested corpus by
                    deterministic string comparison (zero LLM calls)
2. Eval Harness   — labeled-set Hit@K/MRR evaluation + stage-level
                    error diagnosis
3. Golden Runner  — golden tests that catch pipeline output regressions
                    as diffs

Design principle: the verification layer must be deterministic.
Verifying an LLM with another LLM makes the verification itself
probabilistic — and therefore unauditable.
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
