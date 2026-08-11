// CiteGuard web demo engine — a faithful JavaScript port of the Python
// verification logic (citeguard/normalize.py, similarity.py, gate.py,
// metrics.py, golden.py and examples/search_demo/search.py).
//
// Everything here is deterministic and dependency-free, so the demo page
// runs the REAL algorithms client-side. Parity with the Python
// implementation is checked by scripts/check_js_parity.py against the
// shared datasets.

"use strict";

/* ---------------------------------------------------------------- normalize */

const QUOTE_MAP = { "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "·": " " };
const ZERO_WIDTH = /[​‌‍﻿]/g;

function normalize(text) {
  let t = text.normalize("NFKC");
  t = t.replace(/[“”‘’–—·]/g, (ch) => QUOTE_MAP[ch]);
  t = t.replace(ZERO_WIDTH, "");
  t = t.toLowerCase();
  t = t.replace(/\s+/g, " ");
  return t.trim();
}

function squash(text) {
  return normalize(text).replace(/ /g, "");
}

/* --------------------------------------------------------------- similarity */

// Levenshtein edit distance with an optional cap (branch-and-bound early
// exit). Metric properties (row-minimum monotonicity, length lower bound)
// make the early exits result-preserving — see MATH.md §2.
function levenshtein(a, b, cap = null) {
  if (a === b) return 0;
  if (cap !== null && Math.abs(a.length - b.length) > cap) return cap + 1;
  if (a.length < b.length) [a, b] = [b, a];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const curr = new Array(b.length + 1);
    curr[0] = i;
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
      if (curr[j] < rowMin) rowMin = curr[j];
    }
    if (cap !== null && rowMin > cap) return cap + 1;
    prev = curr;
  }
  return prev[b.length];
}

// Two-pass coarse-to-fine minimum-distance window search (exact; see gate.py).
function bestWindow(haystack, needle) {
  const n = needle.length;
  if (n === 0 || !haystack) return { ratio: 0, window: null, index: -1 };
  const last = Math.max(1, haystack.length - n + 1);
  let bestD = null, bestI = 0;
  const step = Math.max(1, Math.floor(n / 4));
  for (let i = 0; i < last; i += step) {
    const cap = bestD === null ? null : bestD - 1;
    const d = levenshtein(needle, haystack.slice(i, i + n), cap);
    if (cap === null || d <= cap) { bestD = d; bestI = i; }
  }
  for (let i = 0; i < last; i++) {
    if (bestD === 0) break;
    const d = levenshtein(needle, haystack.slice(i, i + n), bestD - 1);
    if (d < bestD) { bestD = d; bestI = i; }
  }
  const window = haystack.slice(bestI, bestI + n);
  return { ratio: 1 - bestD / Math.max(n, window.length), window, index: bestI };
}

// Character-level alignment (Wagner–Fischer backtrace) for diff rendering.
// Returns ops: {type: "match"|"sub"|"ins"|"del", a, b}
function alignChars(a, b) {
  const m = a.length, n = b.length;
  const d = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) d[i][0] = i;
  for (let j = 0; j <= n; j++) d[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  const ops = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)) {
      ops.push({ type: a[i - 1] === b[j - 1] ? "match" : "sub", a: a[i - 1], b: b[j - 1] });
      i--; j--;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      ops.push({ type: "del", a: a[i - 1], b: null }); i--;
    } else {
      ops.push({ type: "ins", a: null, b: b[j - 1] }); j--;
    }
  }
  return ops.reverse();
}

/* ------------------------------------------------------------ citation gate */

// corpus: { file: { page(str|int): text } }
function corpusHasFile(corpus, f) { return Object.prototype.hasOwnProperty.call(corpus, f); }
function corpusPage(corpus, f, p) { return corpus[f]?.[String(p)]; }

function findExactInFile(corpus, file, quote, skipPage = null) {
  if (!quote) return null;
  const pages = Object.keys(corpus[file]).map(Number).sort((x, y) => x - y);
  for (const p of pages) {
    if (skipPage !== null && p === Number(skipPage)) continue;
    if (normalize(corpus[file][String(p)]).includes(quote)) return p;
  }
  return null;
}

// Mirrors CitationGate.check — six statuses, deterministic.
function checkCitation(corpus, citation, fuzzyThreshold = 0.85) {
  const quote = normalize(citation.quote);
  const base = { citation, matchedText: null, foundPage: null };

  if (!corpusHasFile(corpus, citation.source_file))
    return { ...base, status: "file_not_found", score: 0, passed: false };

  const pageText = corpusPage(corpus, citation.source_file, citation.page);
  if (pageText === undefined) {
    const found = findExactInFile(corpus, citation.source_file, quote);
    if (found !== null)
      return { ...base, status: "wrong_page", score: 1, matchedText: quote, foundPage: found, passed: false };
    return { ...base, status: "page_not_found", score: 0, passed: false };
  }

  const normPage = normalize(pageText);
  if (quote && normPage.includes(quote))
    return { ...base, status: "verified", score: 1, matchedText: quote, passed: true };

  const { ratio, window } = bestWindow(normPage, quote);
  if (ratio >= fuzzyThreshold)
    return { ...base, status: "fuzzy_match", score: ratio, matchedText: window, passed: true };

  const found = findExactInFile(corpus, citation.source_file, quote, citation.page);
  if (found !== null)
    return { ...base, status: "wrong_page", score: 1, matchedText: quote, foundPage: found, passed: false };

  return { ...base, status: "quote_not_found", score: ratio, matchedText: window, passed: false };
}

/* ------------------------------------------------------------ hybrid search */

const QTY_PATTERN = /(\d+\s*(개|켤레|장|롤|매|입|묶음|박스|세트|자루|병|통|권|ea)|(하나|둘|셋|다섯|열|스무)\s*(개|장|켤레)?|(한|두|세|네|다섯|열)\s*(개|장|켤레|묶음|박스))/gi;
const NOISE_WORDS = ["주세요", "주문", "대여", "부탁", "하나만", "젤", "제일", "싼", "좀"];

function parseQuery(query) {
  let core = query.replace(QTY_PATTERN, " ");
  for (const w of NOISE_WORDS) core = core.split(w).join(" ");
  return core.split(/\s+/).filter(Boolean).join(" ");
}

function charNgrams(text, n = 2) {
  const s = squash(text);
  if (s.length < n) return s ? [s] : [];
  const grams = [];
  for (let i = 0; i <= s.length - n; i++) grams.push(s.slice(i, i + n));
  return grams;
}

function buildEngine(products) {
  // alias index: canonical alias → product ids (insertion order preserved)
  const aliasIndex = new Map();
  for (const p of products) {
    for (const term of [p.name, ...(p.aliases || [])]) {
      const key = squash(term);
      if (!aliasIndex.has(key)) aliasIndex.set(key, []);
      aliasIndex.get(key).push(p.id);
    }
  }
  // bm25 index over char bigrams
  const docTokens = new Map(), docLen = new Map();
  for (const p of products) {
    const text = [p.name, p.spec || "", ...(p.aliases || [])].join(" ");
    const counter = new Map();
    for (const g of charNgrams(text)) counter.set(g, (counter.get(g) || 0) + 1);
    docTokens.set(p.id, counter);
    docLen.set(p.id, [...counter.values()].reduce((a, b) => a + b, 0));
  }
  const avgLen = [...docLen.values()].reduce((a, b) => a + b, 0) / Math.max(docLen.size, 1);
  const df = new Map();
  for (const counter of docTokens.values())
    for (const token of counter.keys()) df.set(token, (df.get(token) || 0) + 1);
  const nDocs = products.length;
  const idf = new Map();
  for (const [token, freq] of df) idf.set(token, Math.log(1 + (nDocs - freq + 0.5) / (freq + 0.5)));

  const K1 = 1.5, B = 0.75;

  function aliasSearch(query) {
    const q = squash(query);
    const scores = new Map();
    for (const [term, ids] of aliasIndex) {
      if (term && q.includes(term)) {
        for (const id of ids) {
          const s = term.length / Math.max(q.length, 1);
          if (!scores.has(id) || s > scores.get(id)) scores.set(id, s);
        }
      }
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1]);
  }

  function bm25Search(query, topK = 20) {
    const scores = new Map();
    for (const token of charNgrams(query)) {
      if (!idf.has(token)) continue;
      const w = idf.get(token);
      for (const [docId, counter] of docTokens) {
        const tf = counter.get(token) || 0;
        if (tf === 0) continue;
        const denom = tf + K1 * (1 - B + (B * docLen.get(docId)) / avgLen);
        scores.set(docId, (scores.get(docId) || 0) + (w * tf * (K1 + 1)) / denom);
      }
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK);
  }

  function fuse(aliasHits, bm25Hits, aliasWeight) {
    const fused = new Map();
    for (const [hits, weight] of [[aliasHits, aliasWeight], [bm25Hits, 1.0]]) {
      if (!hits.length) continue;
      const top = hits[0][1] || 1.0;
      for (const [id, score] of hits) fused.set(id, (fused.get(id) || 0) + (weight * score) / top);
    }
    return fused;
  }

  function run(query, { topK = 5, aliasWeight = 2.0 } = {}) {
    const core = parseQuery(query);
    const aliasHits = aliasSearch(core);
    const bm25Hits = bm25Search(core, 20);
    const pool = [...new Set([...aliasHits.map(([id]) => id), ...bm25Hits.map(([id]) => id)])];
    const fused = fuse(aliasHits, bm25Hits, aliasWeight);
    const reranked = [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK).map(([id]) => id);
    return {
      parsed: core,
      aliasScored: aliasHits,
      bm25Scored: bm25Hits.slice(0, 10),
      fusedScored: [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK),
      pool,
      final: reranked,
    };
  }

  return { run };
}

/* ---------------------------------------------------------- metrics & eval */

function hitAtK(ranked, gold, k) { return ranked.slice(0, k).some((c) => gold.has(c)); }
function reciprocalRank(ranked, gold) {
  for (let i = 0; i < ranked.length; i++) if (gold.has(ranked[i])) return 1 / (i + 1);
  return 0;
}

function evaluate(engine, labelset, { ks = [1, 3, 5], aliasWeight = 2.0 } = {}) {
  const results = labelset.map((row) => {
    const trace = engine.run(row.query, { aliasWeight });
    const gold = new Set(row.gold);
    const hits = Object.fromEntries(ks.map((k) => [k, hitAtK(trace.final, gold, k)]));
    const rr = reciprocalRank(trace.final, gold);
    let lostAt = null;
    if (!hits[Math.max(...ks)]) {
      lostAt = "rank>" + Math.max(...ks);
      for (const [name, cands] of [["retrieve", trace.pool], ["rerank", trace.final]]) {
        if (![...gold].some((g) => cands.includes(g))) { lostAt = name; break; }
      }
    }
    return { query: row.query, gold: row.gold, note: row.note || "", trace, hits, rr, lostAt };
  });
  const total = results.length;
  const metrics = {};
  for (const k of ks) metrics["hit@" + k] = results.filter((r) => r.hits[k]).length / total;
  metrics.mrr = results.reduce((a, r) => a + r.rr, 0) / total;
  return { results, metrics, total };
}

/* ---------------------------------------------------------------- json diff */

function jsonDiff(expected, actual, path = "$") {
  const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  if (isObj(expected) && isObj(actual)) {
    const out = [];
    for (const key of [...new Set([...Object.keys(expected), ...Object.keys(actual)])].sort()) {
      const child = path + "." + key;
      if (!(key in actual)) out.push({ path: child, expected: expected[key], actual: null, kind: "missing" });
      else if (!(key in expected)) out.push({ path: child, expected: null, actual: actual[key], kind: "added" });
      else out.push(...jsonDiff(expected[key], actual[key], child));
    }
    return out;
  }
  if (Array.isArray(expected) && Array.isArray(actual)) {
    const out = [];
    for (let i = 0; i < Math.max(expected.length, actual.length); i++) {
      const child = `${path}[${i}]`;
      if (i >= actual.length) out.push({ path: child, expected: expected[i], actual: null, kind: "missing" });
      else if (i >= expected.length) out.push({ path: child, expected: null, actual: actual[i], kind: "added" });
      else out.push(...jsonDiff(expected[i], actual[i], child));
    }
    return out;
  }
  if (JSON.stringify(expected) !== JSON.stringify(actual))
    return [{ path, expected, actual, kind: "changed" }];
  return [];
}

/* ----------------------------------------------------------------- exports */

const CiteGuard = {
  normalize, squash, levenshtein, bestWindow, alignChars,
  checkCitation, findExactInFile,
  parseQuery, charNgrams, buildEngine,
  hitAtK, reciprocalRank, evaluate, jsonDiff,
};

if (typeof module !== "undefined") module.exports = CiteGuard;
