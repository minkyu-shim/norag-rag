// Plain JavaScript UI for the V2 API (no framework, no build step).
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
  await Promise.all([loadSettings(), loadDocuments()]);
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

  $("ask-button").disabled = true;
  $("ask-button").textContent = "Thinking…";
  try {
    showResult(await api("POST", "/api/ask", { corpus: corpus(), question: $("question").value, sources }));
  } catch (e) {
    showError(e.message);
  } finally {
    $("ask-button").disabled = false;
    $("ask-button").textContent = "Ask";
  }
});

function showError(message) {
  $("result").classList.remove("hidden");
  showBanner($("status-banner"), message, "err");
  ["warnings", "answer", "citations", "retrieved"].forEach((id) => { $(id).innerHTML = ""; });
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

  // Citations: one chip per cited chunk. Clicking opens the chunk.
  $("citations").innerHTML = "";
  for (const c of r.citations) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = `[${c.label}] ${c.source} · lines ${c.start_line}–${c.end_line}`;
    chip.addEventListener("click", () => openChunk(c));
    $("citations").append(chip);
  }

  // All retrieved chunks, cited ones highlighted: the user sees what the LLM was given.
  const cited = new Set(r.citations.map((c) => c.id));
  $("retrieved").innerHTML = "";
  for (const c of r.retrieved) {
    const li = document.createElement("li");
    li.classList.toggle("cited", cited.has(c.id));
    li.innerHTML = `<div><strong></strong> <span class="meta"></span></div><div class="preview"></div>`;
    li.querySelector("strong").textContent = `[${c.label}]${cited.has(c.id) ? " cited" : ""}`;
    li.querySelector(".meta").textContent =
      `${c.source} · lines ${c.start_line}–${c.end_line} · distance ${c.distance.toFixed(3)}`;
    li.querySelector(".preview").textContent = c.text.replace(/\s+/g, " ");
    li.addEventListener("click", () => openChunk(c));
    $("retrieved").append(li);
  }
}

// --- Chunk panel: opening a reference shows the chunk it points to -------------
function openChunk(c) {
  $("chunk-title").textContent = `[${c.label}] ${c.source}`;
  $("chunk-meta").textContent =
    `lines ${c.start_line}–${c.end_line} · characters ${c.start}–${c.end} · distance ${c.distance.toFixed(3)} · id ${c.id}`;
  $("chunk-text").textContent = c.text;
  $("chunk-panel").classList.remove("hidden");
}
$("chunk-close").addEventListener("click", () => $("chunk-panel").classList.add("hidden"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("chunk-panel").classList.add("hidden"); });

// --- Start -------------------------------------------------------------------
// The corpus list comes first: every other call needs a corpus.
loadCorpora().then(refresh).catch((e) => showError(e.message));
