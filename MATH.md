# The Mathematics Behind CiteGuard

This document explains the mathematical structure each module stands on, and
why that structure turns into engineering guarantees — correctness,
performance, auditability — with pointers into the code.

> One-line summary: **the reliability of a verification layer comes from
> structure, not probability.** CiteGuard builds its verdicts on equivalence
> relations, metric spaces, and probabilistic ranking theory, which is what
> guarantees that the same input always produces the same verdict.

---

## 1. Text normalization = equivalence relations & canonical forms

**Where:** `citeguard/normalize.py` · pinned by `tests/test_normalize.py::test_idempotent`

"The same sentence, written differently" has a precise definition: an
**equivalence relation** on the set of strings Σ\*,
`x ~ y ⟺ normalize(x) = normalize(y)`.

- `normalize` is an **idempotent map** π : Σ\* → Σ\* (π∘π = π — pinned by a
  test). The image of an idempotent map is a set of **canonical
  representatives**, one per equivalence class.
- "Does this quote appear in the source?" is really a question about
  equivalence classes (containment up to notation). Projecting both sides to
  canonical form **reduces class-level equality to plain substring matching.**
- Engineering consequence: full-width/half-width variants, smart quotes, and
  whitespace runs all collapse to one representative, so every later
  comparison is deterministic and reproducible.

**Interview one-liner:** "Normalization isn't cosmetic cleanup — it selects a
canonical representative of an equivalence class, so *which* notational
variants count as 'the same sentence' is defined by a relation, not buried in
code."

## 2. Edit distance = a metric on the space of strings

**Where:** `citeguard/similarity.py` · axioms verified by `tests/test_similarity.py::TestMetricAxioms`

The Levenshtein distance d used for fuzzy matching is a **metric**:

| Axiom | Meaning | Why it holds |
|-------|---------|--------------|
| d(a,b) = 0 ⟺ a = b | identity of indiscernibles | zero edits = identical strings |
| d(a,b) = d(b,a) | symmetry | insertions and deletions are mutually inverse |
| d(a,c) ≤ d(a,b) + d(b,c) | **triangle inequality** | concatenate the a→b and b→c edit scripts |

These axioms are not just documented — they are **checked empirically by
parametrized pytest property tests** (symmetry, triangle inequality, and the
length bound over all sample pairs).

The metric structure yields two **provably safe optimizations**
(`gate.py::_best_window`):

1. **Length-bound pruning:** one edit changes length by at most 1, so
   |len(a)−len(b)| ≤ d(a,b). Any window whose length gap already exceeds the
   cap can be skipped **without changing the result.**
2. **Early exit via row-minimum monotonicity:** in the Wagner–Fischer DP
   table, the minimum of row i never decreases in later rows. The moment a
   row minimum exceeds the cap, "d > cap" is final.
   → The two-pass coarse-to-fine window search (coarse scan for a tight
   bound, then an exhaustive stride-1 scan pruned by it) returns **exactly
   the same minimum as an uncapped exhaustive scan**
   (`tests/test_similarity.py::test_cap_never_flips_threshold_decision` pins
   the underlying property).

**Interview one-liner:** "If fuzzy-match pruning were a heuristic, the
verification layer would lose determinism. Because edit distance satisfies
the metric axioms, I can prove the pruned search returns the same result as
exhaustive search — and the axioms themselves are pinned by property tests."

## 3. BM25 = the probability ranking principle & saturation

**Where:** `examples/search_demo/search.py::BM25`

BM25 is not an arbitrary formula — it derives from the **Probability Ranking
Principle** (present documents in decreasing order of relevance probability):

- **IDF term** `log(1 + (N − df + 0.5)/(df + 0.5))` is a smoothed
  Robertson–Spärck Jones weight: the log odds ratio of a token appearing in
  relevant vs non-relevant documents. Information-theoretically, rarer tokens
  carry more self-information (−log p), hence more discriminating power.
- **TF saturation** `tf·(k1+1)/(tf + k1·norm)` is **monotone increasing,
  concave, and bounded** in tf, converging to (k1+1)·idf. Repeated
  occurrences of the same token have diminishing returns — a document
  mentioning "glove" ten times is not ten times more relevant.
- **Length normalization** `(1 − b + b·len/avg_len)` corrects the bias of
  long documents accidentally containing more tokens (b interpolates the
  correction strength).

**Interview one-liner:** "I can explain each of BM25's three factors as an
odds-ratio approximation, a bounded concave saturation, and a length-bias
correction — so I know exactly what tuning k1 and b moves."

## 4. Char n-grams = locality, robustness to typos

**Where:** `examples/search_demo/search.py::char_ngrams`

Splitting a string into character bigrams projects it onto a **multiset of
overlapping local fragments.** A single-character typo destroys at most n
n-grams around it (locality); every other fragment still overlaps. That is
why the typo "빽색" still retrieves "백색" (white) products. For Korean
retrieval without a morphological analyzer, this removes tokenizer error as
a failure mode entirely.

## 5. Ranking metrics = expectations of indicator variables

**Where:** `citeguard/metrics.py`

- **Hit@K** is the sample mean of the indicator 1[gold ∈ Top-K] — an
  unbiased estimate of "the probability a user finds the right answer in the
  top K."
- **MRR** averages reciprocal ranks 1/rank. The reciprocal transform
  concentrates value at the top (dropping from rank 1→2 costs 0.5; from
  9→10 costs 0.011) — a loss structure aligned with Top-1-centric UX.

## 6. Score fusion = scale-invariant weighted combination

**Where:** `examples/search_demo/search.py::HybridSearch._fuse`

The alias channel (dictionary scores) and the BM25 channel (statistical
scores) live in **different units.** Normalizing each channel by its own
maximum projects both into [0,1]; the weighted sum is then **invariant under
positive rescaling of either channel** — retuning BM25 parameters cannot
silently destabilize the fused ranking.

## 7. Golden diff = structural equality of ordered labeled trees

**Where:** `citeguard/golden.py::diff`

JSON values are **inductively defined ordered labeled trees** (leaves =
scalars, internal nodes = objects/arrays). `diff` compares two trees by
**structural recursion** over that inductive definition — and for ordered
trees, structural equality *is* tree isomorphism, so the comparison is a
decision procedure for "are these two runs isomorphic?" Every difference is
reported with its root-to-leaf path (`$.metrics.hit@1`), so a regression
carries a certificate of **exactly which subtree changed.**

---

## Design stance: why no LLM in the verification layer

Verifying an LLM with another LLM makes the verifier itself probabilistic:
`verify(x)` can differ between runs, which makes auditing impossible. Every
verdict function in CiteGuard is a **deterministic function** defined on the
mathematical structures above (equivalence relations, metrics, tree
equality), which buys:

1. Same input → same verdict, always (reproducibility)
2. Verdicts come with evidence — paths, distances, matched spans (explainability)
3. Performance optimizations provably cannot change results (trustworthiness)

That is what "math-grounded design" means here — not formulas as decoration,
but **system guarantees derived from mathematical structure.**
