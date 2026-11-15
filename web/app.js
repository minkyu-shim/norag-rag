// Plain JavaScript UI for the V3 API (no framework, no build step).
// Every call goes through api(), which turns an HTTP error into a readable message.

const $ = (id) => document.getElementById(id);

// The corpus picked in the header. Every API call is about ONE corpus, so it goes in every call.
const corpus = () => $("corpus").value;
const withCorpus = (path) => `${path}${path.includes("?") ? "&" : "?"}corpus=${encodeURIComponent(corpus())}`;

async function api(method, path, body) {
  const options = { method, headers: {} };
  if (body instanceof FormData) {
    options.body = body;                       // file upload: the browser sets the headers
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("Cannot reach the server. Is uvicorn running?");
  }

  const data = await response.json().catch(() => null);
  // /api/ask returns a full AskResponse even on 429/502/504: let the caller show it.
  if (data && data.status) return data;
  if (!response.ok) throw new Error(errorMessage(data, response.status));
  return data;
}

function errorMessage(data, httpStatus) {
  if (!data || !data.detail) return `Server error (HTTP ${httpStatus}).`;
  if (typeof data.detail === "string") return data.detail;
  // 422 from FastAPI/Pydantic: a list of {loc, msg}
  return data.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ");
}

function showBanner(element, text, kind) {
  element.textContent = text;
  element.className = `banner ${kind}`;
}

// Settings (index_ready) and documents (chunk counts) both change after an index/upload/delete.
async function refresh() {
  vocabularyCorpus = null;    // a re-index or an upload changes the chunks: the Words tab reloads on its next opening
  await Promise.all([loadSettings(), loadDocuments(), loadGraphInfo(), loadQuestionSet()]);
}

// --- Corpus ------------------------------------------------------------------
async function loadCorpora() {
  const names = await api("GET", "/api/corpora");
  // Remember the last corpus in this browser only (a convenience: the server keeps no "current corpus").
  let saved = null;
  try { saved = localStorage.getItem("corpus"); } catch { /* storage blocked: start on the first corpus */ }
  for (const name of names) {
    const option = document.createElement("option");
    option.value = option.textContent = name;
    $("corpus").append(option);
  }
  if (names.includes(saved)) $("corpus").value = saved;
}

$("corpus").addEventListener("change", () => {
  try { localStorage.setItem("corpus", corpus()); } catch { /* not critical */ }
  // The documents and the last answer belong to the old corpus: start clean.
  $("source-list").innerHTML = "";
  $("result").classList.add("hidden");
  ["documents-message", "settings-message"].forEach((id) => $(id).classList.add("hidden"));
  refresh().catch((e) => showError(e.message));
});

// --- Tabs --------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
    $(`tab-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

// --- Settings ----------------------------------------------------------------
async function loadSettings() {
  const asked = corpus();
  const s = await api("GET", withCorpus("/api/settings"));
  if (asked !== corpus()) return;   // the user switched corpus meanwhile: this answer is stale
  $("chunk_size").value = s.chunk_size;
  $("overlap_percent").value = s.overlap_percent;
  $("top_k").value = s.top_k;
  showSettingsState(s);
}

function showSettingsState(s) {
  $("index-warning").classList.toggle("hidden", s.index_ready);
  $("settings-info").textContent =
    `Corpus ${s.corpus} — collection ${s.collection} — ${s.index_ready ? "index ready" : "index NOT built"}.`;
}

$("settings-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const s = await api("PUT", withCorpus("/api/settings"), {
      chunk_size: Number($("chunk_size").value),
      overlap_percent: Number($("overlap_percent").value),
      top_k: Number($("top_k").value),
    });
    showSettingsState(s);
    showBanner($("settings-message"), s.index_ready ? "Saved." : "Saved. Re-index to use these chunk settings.",
      s.index_ready ? "ok" : "warn");
    loadDocuments();
  } catch (e) {
    showBanner($("settings-message"), e.message, "err");
  }
});

async function reindex() {
  const buttons = [$("reindex-button"), $("reindex-inline")];
  const labels = buttons.map((b) => b.textContent);   // restored afterwards: the HTML stays the only source
  buttons.forEach((b) => { b.disabled = true; b.textContent = "Indexing…"; });
  try {
    const result = await api("POST", withCorpus("/api/index"));
    const total = Object.values(result.indexed).reduce((a, b) => a + b, 0);
    showBanner($("settings-message"), `Indexed ${total} chunks in ${result.corpus}.`, "ok");
  } catch (e) {
    showBanner($("settings-message"), e.message, "err");
  } finally {
    buttons.forEach((b, i) => { b.disabled = false; b.textContent = labels[i]; });
    await refresh();
  }
}
$("reindex-button").addEventListener("click", reindex);
$("reindex-inline").addEventListener("click", reindex);

// --- Documents ---------------------------------------------------------------
function checkedSources() {
  return [...document.querySelectorAll("#source-list input")].filter((i) => i.checked).map((i) => i.value);
}

async function loadDocuments() {
  const asked = corpus();
  const docs = await api("GET", withCorpus("/api/documents"));
  if (asked !== corpus()) return;   // the user switched corpus meanwhile: this answer is stale

  // Remember which sources were checked, so a refresh does not reset the user's choice.
  const previouslyChecked = new Set(checkedSources());
  const firstLoad = document.querySelectorAll("#source-list input").length === 0;

  $("source-list").innerHTML = "";
  $("documents-table").innerHTML = "";
  for (const doc of docs) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = doc.name;
    checkbox.checked = firstLoad || previouslyChecked.has(doc.name);
    label.append(checkbox, " ", doc.name);
    $("source-list").append(label);

    const row = document.createElement("tr");
    row.innerHTML = `<td></td><td>${(doc.size_bytes / 1024).toFixed(1)} KB</td>
      <td>${doc.chunks || "not indexed"}</td><td><button class="danger">Delete</button></td>`;
    row.cells[0].textContent = doc.name;
    row.querySelector("button").addEventListener("click", () => deleteDocument(doc.name));
    $("documents-table").append(row);
  }
  if (docs.length === 0) $("source-list").textContent = "No documents in this corpus yet: upload one in the Documents tab.";
}

async function deleteDocument(name) {
  if (!confirm(`Delete ${name} from ${corpus()} and all its chunks?`)) return;
  try {
    await api("DELETE", withCorpus(`/api/documents/${encodeURIComponent(name)}`));
    showBanner($("documents-message"), `Deleted ${name}.`, "ok");
  } catch (e) {
    showBanner($("documents-message"), e.message, "err");
  }
  refresh();
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const button = event.submitter;
  button.disabled = true;
  showBanner($("documents-message"), `Uploading and indexing ${file.name} into ${corpus()}…`, "");
  try {
    const result = await api("POST", withCorpus("/api/documents"), form);
    showBanner($("documents-message"), `${result.corpus}/${result.name}: ${result.chunks} chunks indexed.`, "ok");
    $("upload-form").reset();
  } catch (e) {
    showBanner($("documents-message"), e.message, "err");
  } finally {
    button.disabled = false;
  }
  refresh();
});

// --- Retrievers (V3) ----------------------------------------------------------
function checkedRetrievers() {
  return [...document.querySelectorAll("#retriever-list input")].filter((i) => i.checked).map((i) => i.value);
}

// The graph checkbox is only usable when a graph was built for this corpus (python scripts/build_graph.py).
async function loadGraphInfo() {
  const asked = corpus();
  const info = await api("GET", withCorpus("/api/graph"));
  if (asked !== corpus()) return;
  const box = document.querySelector('#retriever-list input[value="graph"]');
  box.disabled = !info.available;
  if (!info.available) box.checked = false;
  $("graph-option").title = info.available
    ? `${info.nodes} entities, ${info.edges} relations (built ${info.built})`
    : "No concept graph for this corpus: run python scripts/build_graph.py " + info.corpus;
}

// "dense #10 · bm25 #1 · RRF 0.0307": which retriever(s) found the chunk, and at which rank in their own list.
function foundByText(c) {
  const parts = Object.entries(c.found_by).map(([name, rank]) => `${name} #${rank}`);
  if (c.rrf_score !== null) parts.push(`RRF ${c.rrf_score.toFixed(4)}`);
  return parts.join(" · ");
}

// Scores are only known for the retrievers that found the chunk.
function scoresText(c) {
  const parts = [];
  if (c.distance !== null) parts.push(`distance ${c.distance.toFixed(3)}`);
  if (c.bm25_score !== null) parts.push(`BM25 ${c.bm25_score.toFixed(2)}`);
  return parts.join(" · ");
}

// --- Ask ---------------------------------------------------------------------
const STATUS_KIND = {
  answered: "ok",
  not_found: "warn",
  unverified: "warn",
  invalid_output: "err",
  provider_error: "err",
};

$("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const sources = checkedSources();
  if (sources.length === 0) {
    showError("Select at least one document to search in.");
    return;
  }
  const retrievers = checkedRetrievers();
  if (retrievers.length === 0) {
    showError("Switch on at least one retriever.");
    return;
  }

  $("ask-button").disabled = true;
  $("ask-button").textContent = "Thinking…";
  try {
    showResult(await api("POST", "/api/ask", { corpus: corpus(), question: $("question").value, sources, retrievers }));
  } catch (e) {
    showError(e.message);
  } finally {
    $("ask-button").disabled = false;
    $("ask-button").textContent = "Ask";
  }
});

function costText(r) {
  const c = r.cost;
  let text = `Retrievers: ${r.retrievers.join(" + ")} · retrieval ${c.retrieval_ms} ms`;
  if (c.llm_calls) text += ` · ${c.llm_calls} LLM call(s), ${c.prompt_tokens} + ${c.completion_tokens} tokens, ${c.llm_ms} ms`;
  return text;
}

function showError(message) {
  $("result").classList.remove("hidden");
  showBanner($("status-banner"), message, "err");
  ["warnings", "answer", "cost", "citations", "retrieved"].forEach((id) => { $(id).innerHTML = ""; });
  $("graph-path").classList.add("hidden");
}

function showResult(r) {
  $("result").classList.remove("hidden");
  showBanner($("status-banner"), `${r.status.replace("_", " ")} — ${r.message}`, STATUS_KIND[r.status] || "");

  $("warnings").innerHTML = "";
  for (const w of r.warnings) {
    const li = document.createElement("li");
    li.textContent = w;
    $("warnings").append(li);
  }

  $("answer").textContent = r.answer || "";
  $("cost").textContent = costText(r);
  showDetailsButton(r);     // V4: opens the trace of this answer (web/trace.js)

  // Citations: one chip per cited chunk. Clicking opens the chunk.
  $("citations").innerHTML = "";
  for (const c of r.citations) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = `[${c.label}] ${c.source} · lines ${c.start_line}–${c.end_line}`;
    chip.addEventListener("click", () => openChunk(c));
    $("citations").append(chip);
  }

  showGraphPath(r);

  // All retrieved chunks, cited ones highlighted: the user sees what the LLM was given.
  const cited = new Set(r.citations.map((c) => c.id));
  $("retrieved").innerHTML = "";
  for (const c of r.retrieved) {
    const li = document.createElement("li");
    li.classList.toggle("cited", cited.has(c.id));
    li.innerHTML = `<div><strong></strong> <span class="meta"></span></div>
      <div class="found-by"></div><div class="preview"></div>`;
    li.querySelector("strong").textContent = `[${c.label}]${cited.has(c.id) ? " cited" : ""}`;
    li.querySelector(".meta").textContent =
      `${c.source} · lines ${c.start_line}–${c.end_line} · ${scoresText(c)}`;
    li.querySelector(".found-by").textContent = `found by: ${foundByText(c)}`;
    li.querySelector(".preview").textContent = c.text.replace(/\s+/g, " ");
    li.addEventListener("click", () => openChunk(c));
    $("retrieved").append(li);
  }
}

// Graph path: entities of the question, then every edge followed (hop by hop) and the chunks of its quote.
// A chunk id that made it into the final top-k is shown with its label (C2); the others are outside the top-k.
function showGraphPath(r) {
  $("graph-path").classList.toggle("hidden", !r.graph_path);
  if (!r.graph_path) return;
  const labels = Object.fromEntries(r.retrieved.map((c) => [c.id, c.label]));
  const p = r.graph_path;
  $("graph-entities").textContent = p.entities.length
    ? `Entities found in the question: ${p.entities.join(", ")}`
    : "No graph entity found in the question: the graph retriever returned nothing.";
  $("graph-edges").innerHTML = "";
  for (const e of p.edges) {
    const li = document.createElement("li");
    const chunks = e.chunk_ids.map((id) => labels[id] || `${id} (not in top-k)`).join(", ") || "no chunk";
    li.textContent = `hop ${e.hop}: ${e.source} —${e.relation}→ ${e.target} · ${chunks}`;
    li.title = `${e.document}: "${e.quote}"`;
    $("graph-edges").append(li);
  }
}

// --- Chunk panel: opening a reference shows the chunk it points to -------------
function openChunk(c) {
  $("chunk-title").textContent = `[${c.label}] ${c.source}`;
  $("chunk-meta").textContent =
    `lines ${c.start_line}–${c.end_line} · characters ${c.start}–${c.end} · ${scoresText(c)} · found by: ${foundByText(c)} · id ${c.id}`;
  $("chunk-text").textContent = c.text;
  $("chunk-panel").classList.remove("hidden");
}
$("chunk-close").addEventListener("click", () => $("chunk-panel").classList.add("hidden"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("chunk-panel").classList.add("hidden"); });

// --- Words analytics (V2) -------------------------------------------------------
// Corpus mode: vocabulary, and word -> chunks. Question mode: question -> words. See app/words.py.

// Wraps the given words in <mark>, using the same token pattern as the BM25 tokenizer (app/bm25.py).
// Built with text nodes, never innerHTML: the chunk text is untrusted.
function highlighted(text, words) {
  const fragment = document.createDocumentFragment();
  const pattern = /\w+(?:[-.]\w+)*/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const token = match[0].toLowerCase();
    const parts = token.split(/[-.]/);
    if (!words.has(token) && !parts.some((p) => words.has(p))) continue;
    fragment.append(text.slice(last, match.index));
    const mark = document.createElement("mark");
    mark.textContent = match[0];
    fragment.append(mark);
    last = match.index + match[0].length;
  }
  fragment.append(text.slice(last));
  return fragment;
}

function openWordChunk(c, words, extra = "") {
  $("chunk-title").textContent = c.source;
  $("chunk-meta").textContent = `lines ${c.start_line}–${c.end_line} · characters ${c.start}–${c.end} · id ${c.id}${extra}`;
  $("chunk-text").replaceChildren(highlighted(c.text, words));
  $("chunk-panel").classList.remove("hidden");
}

function chunkListItem(c, words, label) {
  const li = document.createElement("li");
  li.innerHTML = `<div><strong></strong> <span class="meta"></span></div><div class="preview"></div>`;
  li.querySelector("strong").textContent = label;
  li.querySelector(".meta").textContent = `${c.source} · lines ${c.start_line}–${c.end_line} · ${c.id}`;
  li.querySelector(".preview").textContent = c.text.replace(/\s+/g, " ");
  li.addEventListener("click", () => openWordChunk(c, words));
  return li;
}

let vocabularyCorpus = null;   // the corpus whose vocabulary is on screen (reloaded when it changes)

document.querySelectorAll(".mode").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b === button));
    $("words-corpus").classList.toggle("hidden", button.dataset.mode !== "corpus");
    $("words-question").classList.toggle("hidden", button.dataset.mode !== "question");
  });
});

async function loadVocabulary() {
  const asked = corpus();
  const params = new URLSearchParams({ contains: $("vocab-filter").value.trim(), order: $("vocab-order").value });
  try {
    const v = await api("GET", withCorpus(`/api/words/vocabulary?${params}`));
    if (asked !== corpus()) return;
    vocabularyCorpus = asked;
    $("words-message").classList.add("hidden");
    $("vocab-stats").textContent = `${v.corpus}: ${v.chunks} chunks, ${v.tokens} words in total, ` +
      `${v.distinct_words} distinct words` +
      (v.matching_words !== v.distinct_words ? ` (${v.matching_words} match the filter)` : "") +
      (v.words.length < v.matching_words ? ` — showing the first ${v.words.length}` : "") + ". Click a word to see its chunks.";
    const body = $("vocab-table").querySelector("tbody");
    body.innerHTML = "";
    for (const w of v.words) {
      const row = document.createElement("tr");
      row.innerHTML = `<td><button type="button" class="link"></button></td><td class="num">${w.count}</td><td class="num">${w.chunks}</td>`;
      row.querySelector("button").textContent = w.word;
      row.querySelector("button").addEventListener("click", () => showWord(w.word));
      body.append(row);
    }
    $("vocab-table").classList.remove("hidden");
  } catch (e) {
    showBanner($("words-message"), e.message, "err");
  }
}
$("vocab-form").addEventListener("submit", (event) => { event.preventDefault(); loadVocabulary(); });
$("vocab-order").addEventListener("change", loadVocabulary);

async function showWord(word) {
  $("word-input").value = word;
  try {
    const w = await api("GET", withCorpus(`/api/words/word?word=${encodeURIComponent(word)}`));
    $("words-message").classList.add("hidden");
    $("word-result").classList.remove("hidden");
    $("word-chunks").innerHTML = "";
    if (w.ignored) {
      $("word-title").textContent = w.message;
      return;
    }
    $("word-title").textContent = w.chunk_count
      ? `“${w.word}”: ${w.count} occurrence(s) in ${w.chunk_count} of ${w.total_chunks} chunks (most occurrences first)`
      : `“${w.word}” does not appear anywhere in this corpus.`;
    for (const c of w.chunks) {
      $("word-chunks").append(chunkListItem(c, new Set([w.word]), `${c.occurrences}×`));
    }
    $("word-result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    showBanner($("words-message"), e.message, "err");
  }
}
$("word-form").addEventListener("submit", (event) => { event.preventDefault(); showWord($("word-input").value.trim()); });

async function loadQuestionSet() {
  const asked = corpus();
  const questions = await api("GET", withCorpus("/api/words/questions"));
  if (asked !== corpus()) return;
  const select = $("qwords-pick");
  select.length = 1;      // keep the "type your own" option
  for (const q of questions) {
    const option = document.createElement("option");
    option.value = q.id;
    option.textContent = `${q.id} · ${q.type}${q.answerable ? "" : " (unanswerable)"} · ${q.question}`;
    select.append(option);
  }
}
// Picking a question shows its text; typing a question un-picks the set question.
$("qwords-pick").addEventListener("change", () => {
  const option = $("qwords-pick").selectedOptions[0];
  $("qwords-text").value = $("qwords-pick").value ? option.textContent.split(" · ").slice(2).join(" · ") : "";
});
$("qwords-text").addEventListener("input", () => { $("qwords-pick").value = ""; });

$("qwords-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = { corpus: corpus(), question: $("qwords-text").value, question_id: $("qwords-pick").value || null };
  try {
    showQuestionWords(await api("POST", "/api/words/question", body));
    $("words-message").classList.add("hidden");
  } catch (e) {
    showBanner($("words-message"), e.message, "err");
  }
});

function showQuestionWords(r) {
  $("qwords-result").classList.remove("hidden");
  const fromSet = r.question_id !== null;
  const parts = [`${r.words.length} indexed word(s) in the question, ${r.total_chunks} chunks in the corpus.`];
  if (r.absent.length) parts.push(`Absent from the corpus: ${r.absent.join(", ")}.`);
  if (r.stopwords.length) parts.push(`Stopwords (not indexed): ${r.stopwords.join(", ")}.`);
  if (fromSet && !r.answerable) parts.push("This question is unanswerable: no chunk holds its answer.");
  $("qwords-summary").textContent = parts.join(" ");

  const body = $("qwords-table").querySelector("tbody");
  body.innerHTML = "";
  for (const w of r.words) {
    const row = document.createElement("tr");
    row.classList.toggle("absent-word", w.chunk_count === 0);
    const share = (w.chunk_share * 100).toFixed(1);
    let inEvidence = "–";
    if (fromSet && r.evidence_chunks.length) {
      inEvidence = w.in_evidence.length
        ? `<span class="yes">yes</span> (${w.in_evidence.length}/${r.evidence_chunks.length})`
        : `<span class="no">no</span>`;
    }
    row.innerHTML = `<td><button type="button" class="link"></button></td><td class="num">${w.count}</td>
      <td class="num">${w.chunk_count} (${share}%)</td><td>${inEvidence}</td>`;
    row.querySelector("button").textContent = w.word;
    row.querySelector("button").title = "Show the chunks that contain this word (corpus mode)";
    row.querySelector("button").addEventListener("click", () => {
      document.querySelector('.mode[data-mode="corpus"]').click();
      showWord(w.word);
    });
    body.append(row);
  }

  const words = new Set(r.words.map((w) => w.word));
  $("qwords-evidence").classList.toggle("hidden", r.evidence_chunks.length === 0);
  $("qwords-evidence-list").innerHTML = "";
  for (const c of r.evidence_chunks) {
    $("qwords-evidence-list").append(chunkListItem(c, words, "answer"));
  }
}

// --- Evaluation (V2: started from the UI too) -------------------------------------
$("eval-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const retrievers = [...document.querySelectorAll("#eval-retrievers input")].filter((i) => i.checked).map((i) => i.value);
  if (retrievers.length === 0) {
    showBanner($("eval-message"), "Switch on at least one retriever.", "err");
    return;
  }
  const withLlm = $("eval-llm").checked;
  const button = $("eval-button");
  button.disabled = true;
  button.textContent = "Running…";
  showBanner($("eval-message"), withLlm
    ? "Running every question and generating the answers: this takes a few minutes, keep this page open."
    : "Running every question (retrieval only): a few seconds.", "");
  try {
    const r = await api("POST", "/api/eval", { corpus: corpus(), retrievers, with_llm: withLlm });
    showEvaluation(r);
    showBanner($("eval-message"), `Done. Saved to ${r.file}`, "ok");
  } catch (e) {
    showBanner($("eval-message"), e.message, "err");
  } finally {
    button.disabled = false;
    button.textContent = "Run the evaluation";
  }
});

function showEvaluation(r) {
  $("eval-result").classList.remove("hidden");
  const s = r.settings;
  $("eval-file").textContent = `${r.file} — retrievers ${s.retrievers.join(" + ")}, chunk ${s.chunk_size}, ` +
    `overlap ${s.overlap_percent}%${s.with_llm ? `, answers by ${s.model} (top-${s.top_k})` : ""}`;

  const metrics = [...s.ks.map((k) => `hit@${k}`), ...s.ks.map((k) => `recall@${k}`)];
  // ALL + the question types: the summary entries that have hit@k ("unanswerable" has an "n" but no hit@k).
  const groups = Object.keys(r.summary).filter((name) => r.summary[name][metrics[0]] !== undefined);
  const head = $("eval-table").querySelector("thead");
  head.innerHTML = `<tr><th>question type</th><th class="num">n</th>${metrics.map((m) => `<th class="num">${m}</th>`).join("")}</tr>`;
  const body = $("eval-table").querySelector("tbody");
  body.innerHTML = "";
  for (const name of groups) {
    const line = r.summary[name];
    const row = document.createElement("tr");
    row.classList.toggle("all", name === "ALL");
    row.innerHTML = `<td></td><td class="num">${line.n}</td>${metrics.map((m) => `<td class="num">${line[m].toFixed(2)}</td>`).join("")}`;
    row.cells[0].textContent = name;
    body.append(row);
  }

  const c = r.summary.cost_per_question;
  let extra = `Cost per question: retrieval ${c.retrieval_ms} ms`;
  if (s.with_llm) extra += `, ${c.llm_calls} LLM call(s), ${Math.round(c.prompt_tokens)} + ${Math.round(c.completion_tokens)} tokens, LLM ${c.llm_ms} ms`;
  if (r.summary.unanswerable) {
    const u = r.summary.unanswerable;
    extra += `. Unanswerable: ${u.abstained}/${u.n} abstained, ${u.hallucinated} hallucinated, ${u.errors} errors`;
  }
  $("eval-extra").textContent = extra + ".";

  const qbody = $("eval-questions").querySelector("tbody");
  qbody.innerHTML = "";
  for (const q of r.questions) {
    const row = document.createElement("tr");
    row.innerHTML = "<td></td><td></td><td></td><td></td><td></td>";
    const rank = q.answerable ? (q.rank ?? "missed") : "unanswerable";
    [q.id, q.type, q.question, rank, q.llm_status ?? "–"].forEach((value, i) => { row.cells[i].textContent = value; });
    row.cells[3].className = q.answerable && !q.rank ? "no" : "";
    qbody.append(row);
  }
}

// Clear what belongs to the previous corpus when the corpus changes.
$("corpus").addEventListener("change", () => {
  ["word-result", "qwords-result", "eval-result", "words-message", "eval-message"].forEach((id) => $(id).classList.add("hidden"));
  $("vocab-table").classList.add("hidden");
  $("vocab-stats").textContent = "";
  $("qwords-text").value = "";
  vocabularyCorpus = null;
  if (!$("tab-words").classList.contains("hidden")) loadVocabulary();
});

// The vocabulary is loaded the first time the Words tab is opened (not on page load: nobody may need it).
document.querySelector('.tab[data-tab="words"]').addEventListener("click", () => {
  if (vocabularyCorpus !== corpus()) loadVocabulary();
});

// --- Start -------------------------------------------------------------------
// The corpus list comes first: every other call needs a corpus.
loadCorpora().then(refresh).catch((e) => showError(e.message));
