// CiteGuard demo — UI layer. All computation happens in engine.js (the
// JS port of the Python library, parity-checked in CI).

"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (100 * x).toFixed(1).replace(/\.0$/, "") + "%";

const PRODUCT_BY_ID = Object.fromEntries(PRODUCTS.map((p) => [p.id, p]));
const nameOf = (id) => {
  const p = PRODUCT_BY_ID[id];
  return p ? `<b>${esc(p.name_en)}</b> <span style="color:var(--faint)">${esc(p.name)} · ${id}</span>` : id;
};

const STATUS_LABEL = {
  verified: "VERIFIED",
  fuzzy_match: "FUZZY",
  wrong_page: "WRONG PAGE",
  quote_not_found: "NOT FOUND",
  page_not_found: "NO SUCH PAGE",
  file_not_found: "NO SUCH FILE",
};
const STATUS_COLOR = {
  verified: "var(--ok)", fuzzy_match: "var(--warn)", wrong_page: "var(--info)",
  quote_not_found: "var(--bad)", page_not_found: "var(--bad)", file_not_found: "var(--bad)",
};

/* ==================================================================== */
/* Citation Lab                                                          */
/* ==================================================================== */

const lab = {
  threshold: 0.85,
  citations: CITATIONS_DATA.map((c) => ({ ...c })), // editable copies
  selected: 0,
  activeFile: Object.keys(CORPUS_DATA)[0],
};

function renderCorpus() {
  const tabs = $("#file-tabs");
  tabs.innerHTML = "";
  for (const f of Object.keys(CORPUS_DATA)) {
    const t = el("span", "file-tab" + (f === lab.activeFile ? " active" : ""), esc(f));
    t.onclick = () => { lab.activeFile = f; renderCorpus(); };
    tabs.appendChild(t);
  }
  const pagesBox = $("#corpus-pages");
  pagesBox.innerHTML = "";
  const sel = lab.results?.[lab.selected];
  for (const [page, text] of Object.entries(CORPUS_DATA[lab.activeFile])) {
    const block = el("div", "page-block");
    block.appendChild(el("div", "page-label", `p.${page}`));
    let html = esc(text);
    // Highlight the selected citation's matched span (case-insensitive raw search).
    if (sel && sel.citation.source_file === lab.activeFile) {
      const target = sel.status === "wrong_page" ? String(sel.foundPage) : String(sel.citation.page);
      if (page === target && (sel.status === "verified" || sel.status === "wrong_page")) {
        const idx = text.toLowerCase().indexOf(sel.citation.quote.toLowerCase().trim());
        if (idx >= 0) {
          const q = text.slice(idx, idx + sel.citation.quote.trim().length);
          html = esc(text.slice(0, idx)) + "<mark>" + esc(q) + "</mark>" + esc(text.slice(idx + q.length));
        }
      }
    }
    block.appendChild(el("div", "page-text", html));
    pagesBox.appendChild(block);
  }
}

function charDiffHtml(quote, window) {
  if (!window) return "";
  const cls = { match: "m", sub: "s", ins: "i", del: "d" };
  return CG.alignChars(CG.normalize(quote), window)
    .map((op) => `<span class="${cls[op.type]}">${esc(op.type === "ins" ? op.b : op.a)}</span>`)
    .join("");
}

function detailHtml(r) {
  const s = r.status;
  if (s === "verified")
    return `<div class="detail-note">✓ Quote appears <b>verbatim</b> on the cited page — the highlighted span on the left is the evidence.</div>`;
  if (s === "fuzzy_match")
    return `<div class="detail-label">Character-level diff · quote vs. best source window</div>
            <div class="char-diff">${charDiffHtml(r.citation.quote, r.matchedText)}</div>
            <div class="detail-note" style="margin-top:8px">Similarity <b>${r.score.toFixed(3)}</b> ≥ threshold — accepted as transcription noise. Lower the threshold slider past it to see this flip to NOT FOUND.</div>`;
  if (s === "wrong_page")
    return `<div class="detail-note">The quote is <b>real</b> — but it lives on <b>p.${r.foundPage}</b>, not p.${r.citation.page}. A mislabeled source is a warning, not a pass: the audit trail must be exact.</div>`;
  if (s === "quote_not_found")
    return `<div class="detail-label">Closest source window found (best similarity ${r.score.toFixed(3)})</div>
            <div class="char-diff">${charDiffHtml(r.citation.quote, r.matchedText)}</div>
            <div class="detail-note" style="margin-top:8px">⛔ Nothing in the source supports this sentence — <b>suspected hallucination</b>. This single failure blocks the whole report.</div>`;
  if (s === "page_not_found")
    return `<div class="detail-note">⛔ The cited page does not exist in this file, and the quote appears nowhere else in it.</div>`;
  return `<div class="detail-note">⛔ The cited file was never ingested — the gate refuses to take the report's word for it.</div>`;
}

function runGate() {
  lab.results = lab.citations.map((c) => CG.checkCitation(CORPUS_DATA, c, lab.threshold));
  const allPass = lab.results.every((r) => r.passed);
  const pill = $("#gate-pill");
  pill.textContent = allPass ? "PASS — report may publish" : "BLOCK — report withheld";
  pill.className = "verdict-pill " + (allPass ? "pass" : "block");
  renderCitations();
  renderStrip();
  renderCorpus();
}

function renderCitations() {
  const list = $("#cite-list");
  list.innerHTML = "";
  lab.results.forEach((r, i) => {
    const card = el("div", "cite-card" + (i === lab.selected ? " active" : ""));
    card.onclick = (e) => {
      if (e.target.tagName === "TEXTAREA") return;
      lab.selected = i;
      lab.activeFile = CORPUS_DATA[r.citation.source_file] ? r.citation.source_file : lab.activeFile;
      renderCitations(); renderCorpus(); renderStrip();
    };
    const top = el("div", "cite-top");
    top.appendChild(el("span", "cite-ref", `#${i + 1} · ${esc(r.citation.source_file)} · p.${r.citation.page}`));
    top.appendChild(el("span", `status-chip status-${r.status}`,
      STATUS_LABEL[r.status] + (r.foundPage ? ` → p.${r.foundPage}` : "")));
    card.appendChild(top);

    const ta = el("textarea", "cite-quote");
    ta.value = r.citation.quote;
    ta.rows = Math.max(2, Math.ceil(r.citation.quote.length / 60));
    ta.spellcheck = false;
    ta.oninput = () => { lab.citations[i].quote = ta.value; lab.selected = i; runGateKeepFocus(ta, i); };
    card.appendChild(ta);
    card.appendChild(el("div", "edit-hint", "✎ editable — the verdict recomputes as you type"));

    const track = el("div", "sim-track");
    const fill = el("div", "sim-fill");
    fill.style.width = (r.score * 100).toFixed(1) + "%";
    fill.style.background = STATUS_COLOR[r.status];
    const tick = el("div", "sim-thresh");
    tick.style.left = (lab.threshold * 100).toFixed(1) + "%";
    track.appendChild(fill); track.appendChild(tick);
    card.appendChild(track);
    const meta = el("div", "sim-meta");
    meta.innerHTML = `<span>similarity ${r.score.toFixed(3)}</span><span>threshold ${lab.threshold.toFixed(2)}</span>`;
    card.appendChild(meta);

    const detail = el("div", "cite-detail", detailHtml(r));
    card.appendChild(detail);
    list.appendChild(card);
  });
}

// Re-run the gate while keeping the caret inside the textarea being edited.
function runGateKeepFocus(ta, i) {
  const pos = ta.selectionStart;
  lab.results = lab.citations.map((c) => CG.checkCitation(CORPUS_DATA, c, lab.threshold));
  const allPass = lab.results.every((r) => r.passed);
  const pill = $("#gate-pill");
  pill.textContent = allPass ? "PASS — report may publish" : "BLOCK — report withheld";
  pill.className = "verdict-pill " + (allPass ? "pass" : "block");
  renderCitations(); renderStrip(); renderCorpus();
  const cards = document.querySelectorAll("#cite-list .cite-card");
  const nta = cards[i]?.querySelector("textarea");
  if (nta) { nta.focus(); nta.setSelectionRange(pos, pos); }
}

function renderStrip() {
  const svg = $("#strip");
  const W = svg.clientWidth || 600, H = 74, padX = 14;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (v) => padX + v * (W - 2 * padX);
  let s = `<line x1="${x(0)}" y1="40" x2="${x(1)}" y2="40" stroke="var(--line)" stroke-width="2"/>`;
  for (const t of [0, 0.25, 0.5, 0.75, 1])
    s += `<text x="${x(t)}" y="62" text-anchor="middle" class="axis-label" fill="var(--faint)" font-size="10" font-family="var(--mono)">${t}</text>`;
  const tx = x(lab.threshold);
  s += `<line x1="${tx}" y1="14" x2="${tx}" y2="52" stroke="var(--text)" stroke-dasharray="3 3" opacity="0.7"/>
        <text x="${tx}" y="10" text-anchor="middle" font-size="10" fill="var(--muted)" font-family="var(--mono)">threshold</text>`;
  lab.results.forEach((r, i) => {
    const cx = x(r.score);
    const active = i === lab.selected;
    s += `<circle cx="${cx}" cy="40" r="${active ? 8 : 6}" fill="${STATUS_COLOR[r.status]}" opacity="${active ? 1 : 0.75}" stroke="${active ? "var(--text)" : "none"}" stroke-width="1.5"><title>#${i + 1} ${r.status} · ${r.score.toFixed(3)}</title></circle>`;
  });
  svg.innerHTML = s;
}

$("#thresh").addEventListener("input", (e) => {
  lab.threshold = parseFloat(e.target.value);
  $("#thresh-val").textContent = lab.threshold.toFixed(2);
  runGate();
});

/* --- verify your own document --- */
function runOwn() {
  const src = $("#own-source").value.trim();
  const quote = $("#own-quote").value.trim();
  const box = $("#own-result");
  if (!src || !quote) { box.style.display = "none"; return; }
  const corpus = { "your_document.txt": { "1": src } };
  const r = CG.checkCitation(corpus, { source_file: "your_document.txt", page: 1, quote }, lab.threshold);
  box.style.display = "block";
  box.innerHTML = `
    <div class="cite-card active" style="cursor:default">
      <div class="cite-top">
        <span class="cite-ref">your_document.txt · p.1</span>
        <span class="status-chip status-${r.status}">${STATUS_LABEL[r.status]}</span>
      </div>
      <div class="sim-track"><div class="sim-fill" style="width:${(r.score * 100).toFixed(1)}%;background:${STATUS_COLOR[r.status]}"></div>
        <div class="sim-thresh" style="left:${(lab.threshold * 100).toFixed(1)}%"></div></div>
      <div class="sim-meta"><span>similarity ${r.score.toFixed(3)}</span><span>threshold ${lab.threshold.toFixed(2)}</span></div>
      <div class="cite-detail" style="display:block">${detailHtml(r)}</div>
    </div>`;
}
$("#own-source").addEventListener("input", runOwn);
$("#own-quote").addEventListener("input", runOwn);
$("#own-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => { $("#own-source").value = String(reader.result).slice(0, 20000); runOwn(); };
  reader.readAsText(f);
});

/* ==================================================================== */
/* Search pipeline                                                       */
/* ==================================================================== */

const engine = CG.buildEngine(PRODUCTS);

const EXAMPLES = [
  { q: "데부꾸로 3켤레", gloss: "debukkuro ×3 pairs → work gloves" },
  { q: "빽색 실리콘 2개", gloss: "white (typo) silicone ×2" },
  { q: "베니다 12티 두장", gloss: "veneer 12T ×2 sheets → plywood" },
  { q: "레베루 주세요", gloss: "'leberu' please → spirit level" },
  { q: "가꾸목 다섯개", gloss: "unknown slang → intentional failure" },
];

function scoreRows(container, scored, max) {
  container.innerHTML = "";
  if (!scored.length) { container.appendChild(el("div", "empty-note", "no hits")); return; }
  const top = max ?? (scored[0]?.[1] || 1);
  for (const [id, score] of scored.slice(0, 5)) {
    const row = el("div", "score-row");
    row.appendChild(el("span", "score-name", nameOf(id)));
    const bar = el("div", "score-bar");
    const fill = el("i");
    bar.appendChild(fill);
    row.appendChild(bar);
    row.appendChild(el("span", "score-val", score.toFixed(2)));
    container.appendChild(row);
    requestAnimationFrame(() => { fill.style.width = Math.max(4, (score / top) * 100) + "%"; });
  }
}

function runSearch() {
  const q = $("#query").value;
  if (!q.trim()) return;
  const t = engine.run(q);
  $("#parse-out").innerHTML = `<span style="color:var(--faint)">"${esc(q)}"</span> → <b>"${esc(t.parsed)}"</b>`;
  scoreRows($("#alias-out"), t.aliasScored);
  scoreRows($("#bm25-out"), t.bm25Scored);
  const fusedBox = $("#fused-out");
  fusedBox.innerHTML = "";
  if (t.fusedScored.length) {
    const top = t.fusedScored[0][1] || 1;
    t.fusedScored.forEach(([id, score], i) => {
      const row = el("div", "score-row");
      row.appendChild(el("span", "score-name", `<span style="color:var(--faint);font-family:var(--mono)">${i + 1}</span>&nbsp; ` + nameOf(id)));
      const bar = el("div", "score-bar" + (i === 0 ? " ok" : ""));
      const fill = el("i");
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("span", "score-val", score.toFixed(2)));
      fusedBox.appendChild(row);
      requestAnimationFrame(() => { fill.style.width = Math.max(4, (score / top) * 100) + "%"; });
    });
  } else {
    fusedBox.appendChild(el("div", "empty-note", "no candidates"));
  }
  $("#pool-note").textContent = `union candidate pool: ${t.pool.length} product(s) — a hit in either channel survives`;
  $("#lost-banner").style.display = t.final.length ? "none" : "block";
}

$("#query").addEventListener("input", runSearch);
const chipBox = $("#chips");
for (const ex of EXAMPLES) {
  const c = el("span", "chip", `${esc(ex.q)} <small>· ${esc(ex.gloss)}</small>`);
  c.onclick = () => { $("#query").value = ex.q; runSearch(); };
  chipBox.appendChild(c);
}

/* --- eval --- */
$("#run-eval").addEventListener("click", () => {
  const out = $("#eval-out");
  const rep = CG.evaluate(engine, LABELSET);
  out.style.display = "block";
  $("#eval-tiles").innerHTML = [1, 3, 5].map((k) =>
    `<div class="tile"><div class="t-label">Hit@${k}</div><div class="t-value">${pct(rep.metrics["hit@" + k])}</div></div>`
  ).join("") + `<div class="tile"><div class="t-label">MRR</div><div class="t-value">${rep.metrics.mrr.toFixed(3)}</div></div>`;

  const failures = rep.results.filter((r) => r.lostAt);
  $("#eval-diag").innerHTML = failures.length
    ? failures.map((r) =>
        `<div class="diag-card">Lost at <code>${r.lostAt}</code> — <span class="mono">"${esc(r.query)}"</span>.
         ${r.lostAt === "retrieve"
           ? "A <b>recall</b> problem: the ontology doesn't know this slang yet → alias enrichment, not prompt tuning."
           : "A <b>ranking</b> problem: retrieved but ranked out → fix fusion/re-ranking, not the index."}</div>`
      ).join("")
    : `<div class="diag-card" style="border-color:var(--ok);background:var(--ok-dim)">No failing queries.</div>`;

  $("#eval-table").innerHTML =
    `<tr><th>query</th><th>gold</th><th>top-1</th><th>RR</th><th>lost at</th></tr>` +
    rep.results.map((r) =>
      `<tr class="${r.lostAt ? "fail" : ""}">
         <td class="q">${esc(r.query)}</td>
         <td>${r.gold.map(nameOf).join(", ")}</td>
         <td>${r.trace.final[0] ? nameOf(r.trace.final[0]) : "—"}</td>
         <td class="mono">${r.rr.toFixed(2)}</td>
         <td class="mono">${r.lostAt || "✓"}</td>
       </tr>`
    ).join("");
  out.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

/* ==================================================================== */
/* Golden watch                                                          */
/* ==================================================================== */

// Precompute the weight sweep once — 17 weights × 16 queries, instant.
const SWEEP = [];
for (let w = 0; w <= 4.0001; w += 0.25) {
  const rep = CG.evaluate(engine, LABELSET, { aliasWeight: w });
  const rankings = Object.fromEntries(rep.results.map((r) => [r.query, r.trace.final]));
  SWEEP.push({ w, hit1: rep.metrics["hit@1"], mrr: rep.metrics.mrr, rankings });
}

function renderSweep(current) {
  const svg = $("#sweep");
  const W = svg.clientWidth || 520, H = 220, padL = 40, padR = 14, padT = 16, padB = 28;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (w) => padL + (w / 4) * (W - padL - padR);
  const y = (v) => padT + (1 - v) * (H - padT - padB);
  let s = "";
  for (const g of [0.25, 0.5, 0.75, 1])
    s += `<line x1="${padL}" y1="${y(g)}" x2="${W - padR}" y2="${y(g)}" stroke="var(--line-soft)"/>
          <text x="${padL - 8}" y="${y(g) + 3}" text-anchor="end" class="axis-label">${g}</text>`;
  for (const w of [0, 1, 2, 3, 4])
    s += `<text x="${x(w)}" y="${H - 8}" text-anchor="middle" class="axis-label">${w}</text>`;
  const line = (key, color) =>
    `<polyline fill="none" stroke="${color}" stroke-width="2" points="${SWEEP.map((p) => `${x(p.w)},${y(p[key])}`).join(" ")}"/>`;
  s += line("hit1", "var(--brand)") + line("mrr", "var(--info)");
  s += `<text x="${W - padR}" y="${y(SWEEP[SWEEP.length - 1].hit1) - 8}" text-anchor="end" class="axis-label" fill="var(--brand)">Hit@1</text>`;
  s += `<text x="${W - padR}" y="${y(SWEEP[SWEEP.length - 1].mrr) + 14}" text-anchor="end" class="axis-label" fill="var(--info)">MRR</text>`;
  const cx = x(current);
  s += `<line x1="${cx}" y1="${padT}" x2="${cx}" y2="${H - padB}" stroke="var(--text)" stroke-dasharray="3 3" opacity="0.6"/>`;
  const pt = SWEEP.find((p) => Math.abs(p.w - current) < 1e-6);
  if (pt) s += `<circle cx="${cx}" cy="${y(pt.hit1)}" r="5" fill="var(--brand)" stroke="var(--bg)" stroke-width="2"/>
                <circle cx="${cx}" cy="${y(pt.mrr)}" r="5" fill="var(--info)" stroke="var(--bg)" stroke-width="2"/>`;
  svg.innerHTML = s;
}

function runGolden() {
  const w = parseFloat($("#weight").value);
  $("#weight-val").textContent = w.toFixed(2);
  const pt = SWEEP.find((p) => Math.abs(p.w - w) < 1e-6);
  $("#g-hit1").textContent = pct(pt.hit1);
  $("#g-mrr").textContent = pt.mrr.toFixed(3);
  renderSweep(w);

  // Compare what the baseline pins: the final rankings (the sensitive part).
  const diffs = CG.jsonDiff(GOLDEN_BASELINE.final_rankings, pt.rankings, "$.final_rankings");
  const pill = $("#golden-pill");
  const out = $("#golden-out");
  if (!diffs.length) {
    pill.textContent = "PASS"; pill.className = "verdict-pill pass";
    out.innerHTML = `<div class="golden-ok">✓ Identical to the committed baseline — not a single ranking moved.</div>
      <div style="font-size:13px;color:var(--faint)">Try weight ≠ 2.00: aggregate metrics may barely move, but the diff will name every ranking that shifted. If a change is intentional, the baseline is refreshed only by explicit approval (<code>CITEGUARD_UPDATE_GOLDEN=1</code>).</div>`;
  } else {
    pill.textContent = `FAIL · ${diffs.length} diff${diffs.length > 1 ? "s" : ""}`;
    pill.className = "verdict-pill block";
    out.innerHTML = `<div style="font-size:13px;color:var(--muted);margin-bottom:8px">
        <b style="color:var(--bad)">${diffs.length} ranking difference(s)</b> vs. baseline — exact path, old value, new value:</div>
      <div class="diff-list">` +
      diffs.slice(0, 40).map((d) =>
        `<div class="diff-row"><span class="p">${esc(d.path)}</span><br>
         <span class="from">${esc(JSON.stringify(d.expected))}</span> → <span class="to">${esc(JSON.stringify(d.actual))}</span></div>`
      ).join("") +
      (diffs.length > 40 ? `<div class="diff-row">… and ${diffs.length - 40} more</div>` : "") +
      `</div>`;
  }
}
$("#weight").addEventListener("input", runGolden);

/* ==================================================================== */
/* Scroll reveal + init                                                  */
/* ==================================================================== */

const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
}, { threshold: 0.08 });
document.querySelectorAll(".reveal").forEach((n) => io.observe(n));

runGate();
runSearch();
runGolden();
window.addEventListener("resize", () => { renderStrip(); renderSweep(parseFloat($("#weight").value)); });
