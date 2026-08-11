// Browser smoke test for the web demo (docs/index.html) using jsdom.
//
//   node scripts/check_demo_page.js
//
// Loads the real page with scripts enabled and fails on ANY runtime error —
// exactly the class of bug a syntax check cannot catch (undefined globals,
// missing DOM ids, init-order crashes). Then asserts the page actually
// rendered: citation cards, search results, and golden verdict all present.
//
// jsdom is resolved from CITEGUARD_JSDOM_DIR if set (CI installs it in a
// temp dir), else from normal module resolution.

"use strict";

const path = require("path");
const jsdomDir = process.env.CITEGUARD_JSDOM_DIR;
const { JSDOM } = require(jsdomDir ? path.join(jsdomDir, "node_modules", "jsdom") : "jsdom");

const root = path.join(__dirname, "..");
const errors = [];

const dom = JSDOM.fromFile(path.join(root, "docs", "index.html"), {
  runScripts: "dangerously",
  resources: "usable",
  pretendToBeVisual: true,
});

dom.then((page) => {
  page.window.addEventListener("error", (e) => errors.push(e.message));
  // give external scripts + init a moment to run
  setTimeout(() => {
    const doc = page.window.document;
    const checks = [
      ["citation cards rendered", doc.querySelectorAll("#cite-list .cite-card").length >= 6],
      ["gate verdict shown", /BLOCK|PASS/.test(doc.querySelector("#gate-pill")?.textContent || "")],
      ["search results rendered", doc.querySelectorAll("#fused-out .score-row").length >= 1],
      ["corpus pages rendered", doc.querySelectorAll("#corpus-pages .page-block").length >= 1],
      ["golden verdict shown", /PASS|FAIL/.test(doc.querySelector("#golden-pill")?.textContent || "")],
      ["sweep chart rendered", (doc.querySelector("#sweep")?.innerHTML || "").includes("polyline")],
    ];
    let failed = false;
    for (const [name, ok] of checks) {
      if (!ok) { failed = true; console.error("FAIL:", name); }
    }
    if (errors.length) { failed = true; console.error("Runtime errors:", errors); }
    console.log(failed ? "demo page smoke test: FAILED" : "demo page smoke test: OK");
    process.exit(failed ? 1 : 0);
  }, 800);
}).catch((e) => { console.error(e); process.exit(1); });
