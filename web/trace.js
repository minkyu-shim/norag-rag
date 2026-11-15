// V4 Part A — the "Details" button under an answer, and the side panel that shows its trace.
// A trace = one request; a span = one step of it (a retriever, the fusion, the LLM call...).
// Everything is built with textContent, never innerHTML: a trace holds question and answer text.
// Uses $ and api() from app.js (loaded after this file; they are only called when the user clicks).

const STEP_LABELS = {
  "ask": "Question",
  "check_ready": "Checking the index",
  "retrieve.dense": "Meaning search (dense)",
  "retrieve.bm25": "Keyword search (BM25)",
  "retrieve.graph": "Concept graph search",
  "fusion.rrf": "Merging the results (RRF)",
  "prompt.build": "Building the prompt",
  "llm.call": "Asking the LLM",
  "llm.validate": "Checking the LLM output",
  "verify.citations": "Checking the citations",
  "graph.explain": "Graph path",
  "index.document": "Indexing a document",
  "index.corpus": "Indexing a corpus",
  "index.graph": "Building the concept graph",
  "index.clear_old": "Removing the old chunks",
  "index.chunk": "Cutting into chunks",
  "index.embed_and_store": "Embeddings → vector DB",
  "index.bm25": "Keyword index (BM25)",
  "graph.extract": "Extracting concepts from a chunk",
};

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function attrText(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function usdText(trace) {
  const usd = trace.spans.reduce((sum, s) => sum + (typeof s.attrs.cost_usd === "number" ? s.attrs.cost_usd : 0), 0);
  const priced = trace.spans.some((s) => typeof s.attrs.cost_usd === "number");
  return priced ? `$${usd.toFixed(6)}` : "n/a (set the prices in app/config.py)";
}

// The button under every answer. Called by showResult() in app.js.
function showDetailsButton(r) {
  let button = $("details-button");
  if (!button) {
    button = el("button", "link", "Details");
    button.id = "details-button";
    button.type = "button";
    $("cost").after(button);
  }
  button.classList.toggle("hidden", !r.trace_id);
  button.onclick = () => openTrace(r.trace_id);
}

async function openTrace(traceId) {
  const panel = $("trace-panel");
  panel.classList.remove("hidden");
  $("chunk-panel").classList.add("hidden");
  $("trace-title").textContent = "Details";
  $("trace-summary").textContent = "Loading…";
  $("trace-steps").innerHTML = "";
  try {
    renderTrace(await api("GET", `/api/traces/${encodeURIComponent(traceId)}`));
  } catch (e) {
    $("trace-summary").textContent = e.message;
  }
}

function renderTrace(trace) {
  const a = trace.attrs;
  $("trace-title").textContent = STEP_LABELS[trace.name] || trace.name;

  // Summary: what a first-time user needs. Question, outcome, time, tokens, cost.
  const summary = $("trace-summary");
  summary.innerHTML = "";
  const rows = [
    ["Outcome", (a.outcome || trace.status).replace("_", " ")],
    ["Total time", `${trace.duration_ms} ms`],
    ["LLM", `${a.llm_calls ?? trace.spans.filter((s) => s.name === "llm.call").length} call(s), ` +
      `${a.prompt_tokens ?? 0} + ${a.completion_tokens ?? 0} tokens`],
    ["Cost", usdText(trace)],
    ["Personal data hidden", Object.keys(a.redactions || {}).length
      ? Object.entries(a.redactions).map(([k, n]) => `${n} ${k.toLowerCase()}`).join(", ")
      : "none found"],
  ];
  if (a.question) rows.unshift(["Question", a.question]);
  for (const [k, v] of rows) {
    const line = el("div", "trace-row");
    line.append(el("span", "trace-key", k), el("span", "", v));
    summary.append(line);
  }
  if (trace.status === "error") summary.append(el("p", "banner err", a.error || a.message || "This request ended in an error."));

  // Steps: one row per span, indented by depth, with a bar showing when it ran and how long.
  const total = Math.max(trace.duration_ms || 1, 1);
  const depth = {};
  const byId = Object.fromEntries(trace.spans.map((s) => [s.id, s]));
  const depthOf = (s) => {
    if (depth[s.id] !== undefined) return depth[s.id];
    depth[s.id] = s.parent_id && byId[s.parent_id] ? depthOf(byId[s.parent_id]) + 1 : 0;
    return depth[s.id];
  };

  const steps = $("trace-steps");
  steps.innerHTML = "";
  for (const s of trace.spans) {
    const item = el("details", `trace-step${s.status === "error" ? " failed" : ""}`);
    item.style.marginLeft = `${depthOf(s) * 14}px`;
    const head = el("summary");
    head.append(el("span", "trace-name", STEP_LABELS[s.name] || s.name), el("span", "meta", ` ${s.duration_ms} ms`));
    const bar = el("div", "trace-bar");
    const fill = el("div", "trace-fill");
    fill.style.marginLeft = `${Math.min((s.offset_ms / total) * 100, 99)}%`;
    fill.style.width = `${Math.max(((s.duration_ms || 0) / total) * 100, 1)}%`;
    bar.append(fill);
    head.append(bar);
    item.append(head);
    const list = el("dl", "trace-attrs");
    for (const [k, v] of Object.entries(s.attrs)) list.append(el("dt", "", k), el("dd", "", attrText(v)));
    item.append(list);
    steps.append(item);
  }
}

$("trace-close").addEventListener("click", () => $("trace-panel").classList.add("hidden"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("trace-panel").classList.add("hidden"); });
