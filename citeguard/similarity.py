"""String similarity based on Levenshtein edit distance.

Mathematical background
-----------------------
Edit distance d(a, b) — the minimum number of insertions, deletions, and
substitutions turning a into b — is a **metric** on the set of strings:

  1. d(a, b) ≥ 0,  d(a, b) = 0 ⟺ a = b   (identity of indiscernibles)
  2. d(a, b) = d(b, a)                     (symmetry — edits are invertible)
  3. d(a, c) ≤ d(a, b) + d(b, c)           (triangle inequality — concatenate
                                            the a→b and b→c edit scripts)

Practical consequences of the metric structure:
  · Length lower bound:  |len(a) − len(b)| ≤ d(a, b)
    (one edit changes length by at most 1) → comparisons whose length gap
    already exceeds the cap can be skipped without computing anything.
  · Row-minimum monotonicity: in the Wagner–Fischer table, the minimum of
    row i never decreases in later rows → the moment a row minimum exceeds
    the cap, "d > cap" is already final.

Thanks to these two properties, the cap-based early exit below is a
**result-preserving optimization, not a heuristic** — it never breaks the
determinism of the verification layer.

Algorithm: Wagner–Fischer dynamic programming.
  d[i][j] = edit distance between a[:i] and b[:j]
  d[i][j] = min( d[i-1][j] + 1,            # delete a[i]
                 d[i][j-1] + 1,            # insert b[j]
                 d[i-1][j-1] + (a[i]≠b[j]) # substitute or match )
  Space O(min(n, m)) — only the previous row is kept.
"""

from __future__ import annotations


def levenshtein(a: str, b: str, cap: int | None = None) -> int:
    """Edit distance between a and b. With a cap, returns cap+1 once d > cap.

    Returning cap+1 means "unknown exactly, but exceeds the cap" — which
    loses nothing for threshold decisions (is d ≤ cap?).
    """
    if a == b:
        return 0

    # Skip computation entirely via the length lower bound |len(a)-len(b)| ≤ d
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1

    # Space saving: make the shorter string the column
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        row_min = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
            if curr[j] < row_min:
                row_min = curr[j]
        # Row minimums are monotone non-decreasing → safe early-exit point
        if cap is not None and row_min > cap:
            return cap + 1
        prev = curr
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """Normalized similarity = 1 − d(a,b) / max(|a|,|b|) ∈ [0, 1].

    Dividing by the longer length (an upper bound on d) keeps the value
    in [0, 1]. 1.0 ⟺ exact match.
    """
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))
