// V4 Part B — Concept map explorer (the Concept map tab). API: GET /api/concepts, GET /api/concept.
// Uses the helpers of app.js ($, api, withCorpus, corpus, showBanner, openWordChunk, chunkListItem).
//
//   list of concepts (left)  ->  click one  ->  its neighbourhood drawn as SVG (click a neighbour to move),
//                                               its links with the passage each one was read from,
//                                               the documents that mention it,
//                                               "Ask about this" (fills the Ask tab)
//
// Every text that comes from the documents is put in the page with textContent, never innerHTML.

const SVG_NS = "http://www.w3.org/2000/svg";
const map = {
  corpus: null,       // the corpus whose concept list is loaded (reloaded when the corpus changes)
  concepts: [],       // the full list, filtered in the page by the search box
  current: null,      // the concept on screen (the API response)
  history: [],        // concept ids visited before the current one: the Back button
};

// Relation names as the graph stores them -> words for a reader who does not know the schema.
const RELATION_WORDS = {
  owned_by: "owned by", escalate_to: "escalate to", raised_by: "raised by",
  on_call_channel: "on-call channel", lead: "lead", includes: "includes", replaced: "replaced",
};
const relationWords = (r) => RELATION_WORDS[r] || r.replaceAll("_", " ");

const TYPE_WORDS = { ErrorCode: "Error code", Service: "Service", Team: "Team", Person: "Person",
  Channel: "Channel", Plan: "Plan", Feature: "Feature" };

// A question that makes sense for each type of concept: the user can edit it before asking.
const QUESTION_FOR = {
  ErrorCode: (n) => `What does error ${n} mean, what should support do, and who handles it?`,
  Service: (n) => `What does ${n} do, and which team owns it?`,
  Team: (n) => `What does the ${n} team own, and how can support reach it?`,
  Person: (n) => `What is the role of ${n}?`,
  Channel: (n) => `Which team uses ${n}, and when should support post there?`,
};
const questionAbout = (c) => (QUESTION_FOR[c.type] || ((n) => `What do the documents say about ${n}?`))(c.name);

// The words to highlight in a passage: the names of the concept (and of the neighbour), as tokens.
function wordsOf(...concepts) {
  const words = new Set();
  for (const c of concepts) {
    for (const name of [c.name, ...c.aliases]) {
      for (const m of name.toLowerCase().matchAll(/\w+(?:[-.]\w+)*/g)) words.add(m[0]);
    }
  }
  return words;
}

function costLine(cost) {
  return `Looked up in ${cost.ms} ms · no LLM call`;
}

// --- The list of concepts --------------------------------------------------------
async function loadConceptList() {
  const asked = corpus();
  map.corpus = asked;
  map.current = null;
  map.history = [];
  $("map-message").classList.add("hidden");
  $("map-body").classList.add("hidden");
  $("map-count").textContent = "Loading…";
  let data;
  try {
    data = await api("GET", withCorpus("/api/concepts"));
  } catch (e) {
    if (asked !== corpus()) return;
    map.corpus = null;           // try again on the next opening (the graph may be built by then)
    showBanner($("map-message"), e.message, "warn");
    return;
  }
  if (asked !== corpus()) return;
  map.concepts = data.concepts;
  $("map-body").classList.remove("hidden");
  showConceptState();
  renderConceptList();
}

function renderConceptList() {
  const q = $("map-search").value.trim().toLowerCase();
  const shown = map.concepts.filter((c) => !q || c.type.toLowerCase().includes(q)
    || c.aliases.some((a) => a.toLowerCase().includes(q)));
  $("map-count").textContent = q
    ? `${shown.length} of ${map.concepts.length} concepts`
    : `${map.concepts.length} concepts`;

  const list = $("map-list");
  list.replaceChildren();
  if (!shown.length) {
    const p = document.createElement("p");
    p.className = "meta";
    p.textContent = "No concept matches. Try a team, a service or an error code (KST-…).";
    list.append(p);
    return;
  }
  let type = null;
  for (const c of shown) {
    if (c.type !== type) {
      type = c.type;
      const h = document.createElement("h4");
      h.textContent = TYPE_WORDS[type] || type;
      list.append(h);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `concept t-${c.type}`;
    button.classList.toggle("active", map.current?.concept.id === c.id);
    button.dataset.id = c.id;
    const name = document.createElement("span");
    name.textContent = c.name;
    const links = document.createElement("small");
    links.textContent = c.links === 1 ? "1 link" : `${c.links} links`;
    button.append(name, links);
    button.addEventListener("click", () => showConcept(c.id));
    list.append(button);
  }
}

$("map-search").addEventListener("input", renderConceptList);
$("map-example").addEventListener("click", () => {
  const example = map.concepts.find((c) => c.name === "postroom") || map.concepts[0];
  if (example) showConcept(example.id);
});

// --- One concept ---------------------------------------------------------------
async function showConcept(id, { back = false } = {}) {
  const asked = corpus();
  $("map-cost").textContent = "Loading…";
  let data;
  try {
    data = await api("GET", withCorpus(`/api/concept?id=${encodeURIComponent(id)}`));
  } catch (e) {
    if (asked !== corpus()) return;
    $("map-cost").textContent = "";
    showBanner($("map-message"), e.message, "err");
    return;
  }
  if (asked !== corpus()) return;
  $("map-message").classList.add("hidden");
  if (!back && map.current && map.current.concept.id !== id) map.history.push(map.current.concept.id);
  map.current = data;
  showConceptState();
  renderConceptList();          // marks the active concept in the list

  const c = data.concept;
  $("map-title").textContent = c.name;
  $("map-type").textContent = TYPE_WORDS[c.type] || c.type;
  $("map-type").className = `type-pill t-${c.type}`;
  $("map-back").classList.toggle("hidden", !map.history.length);
  $("map-cost").textContent = costLine(data.cost);
  drawNeighbourhood(data);
  renderLinks(data);
  renderMentions(data);
}

function showConceptState() {
  $("map-empty").classList.toggle("hidden", !!map.current);
  $("map-concept").classList.toggle("hidden", !map.current);
}

$("map-back").addEventListener("click", () => {
  const previous = map.history.pop();
  if (previous) showConcept(previous, { back: true });
});

// --- The drawing: the concept in the middle, its neighbours around it ---------------
function svg(tag, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attributes)) element.setAttribute(k, v);
  return element;
}

function drawNeighbourhood(data) {
  const box = $("map-svg");
  box.replaceChildren();
  const defs = svg("defs");
  const marker = svg("marker", { id: "map-arrow", viewBox: "0 0 10 10", refX: "10", refY: "5",
    markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
  marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow" }));
  defs.append(marker);
  box.append(defs);

  // One node per neighbour, even when there are several links to it (KST-5031 escalate_to AND raised_by postroom).
  const neighbours = new Map();
  for (const link of data.links) {
    const n = neighbours.get(link.other.id) || { concept: link.other, links: [] };
    n.links.push(link);
    neighbours.set(link.other.id, n);
  }

  const centre = data.concept;
  const R = { centre: 30, other: 14 };
  const rx = 165, ry = 105;              // ellipse of the neighbours: the box is wider than tall
  const list = [...neighbours.values()];
  // No neighbour straight below the centre: its edge label would sit on the centre's name.
  // Of the two layouts (first neighbour at the top, or half a step further), keep the one farthest from "below".
  const step = 360 / Math.max(list.length, 1);
  const distanceToBelow = (offset) => Math.min(...list.map((_, i) => {
    const a = (((-90 + offset + i * step) % 360) + 360) % 360;
    return Math.min(Math.abs(a - 90), 360 - Math.abs(a - 90));
  }));
  const offset = distanceToBelow(0) >= distanceToBelow(step / 2) ? 0 : step / 2;
  list.forEach((n, i) => {
    const angle = ((-90 + offset + i * step) * Math.PI) / 180;
    n.x = Math.round(rx * Math.cos(angle));
    n.y = Math.round(ry * Math.sin(angle));
  });

  // Edges first, so that the nodes are drawn on top of them.
  for (const n of list) {
    const length = Math.hypot(n.x, n.y);
    const ux = n.x / length, uy = n.y / length;
    const directions = new Set(n.links.map((l) => l.direction));
    const line = svg("line", {
      x1: ux * R.centre, y1: uy * R.centre, x2: n.x - ux * R.other, y2: n.y - uy * R.other, class: "edge",
    });
    // out: centre --relation--> neighbour (arrow at the neighbour); in: the other way round.
    if (directions.has("out")) line.setAttribute("marker-end", "url(#map-arrow)");
    if (directions.has("in")) line.setAttribute("marker-start", "url(#map-arrow)");
    box.append(line);

    // Beside the middle of the edge, on its outer side (right of an edge going right, left of one going left):
    // on the edge itself it would cross the arrow, and near the centre it would sit on the centre's name.
    const toLeft = n.x < -15;
    const label = svg("text", { x: n.x * 0.5 + (toLeft ? -8 : 8), y: n.y * 0.5 + 4, class: "edge-label",
      "text-anchor": toLeft ? "end" : "start" });
    label.textContent = [...new Set(n.links.map((l) => relationWords(l.relation)))].join(" · ");
    box.append(label);
  }

  const centreNode = svg("g", { class: `node centre t-${centre.type}` });
  centreNode.append(svg("circle", { r: R.centre }));
  const centreName = svg("text", { y: R.centre + 18, "text-anchor": "middle", class: "node-label" });
  centreName.textContent = centre.name;
  centreNode.append(centreName);
  box.append(centreNode);

  for (const n of list) {
    const g = svg("g", { class: `node t-${n.concept.type}`, transform: `translate(${n.x},${n.y})`,
      tabindex: "0", role: "button" });
    const title = svg("title");
    title.textContent = `${n.concept.name} · ${TYPE_WORDS[n.concept.type] || n.concept.type} · `
      + `${n.concept.links} link(s) · click to explore`;
    g.append(title, svg("circle", { r: R.other }));
    // The name goes outwards: on the right of a node on the right, on the left of a node on the left.
    const side = n.x > 20 ? "start" : n.x < -20 ? "end" : "middle";
    const dx = side === "start" ? R.other + 6 : side === "end" ? -(R.other + 6) : 0;
    const dy = side === "middle" ? (n.y < 0 ? -(R.other + 8) : R.other + 16) : 5;
    const name = svg("text", { x: dx, y: dy, "text-anchor": side, class: "node-label" });
    // Long names are shortened in the drawing (the full name is in the list below and in the tooltip).
    name.textContent = n.concept.name.length > 22 ? `${n.concept.name.slice(0, 21)}…` : n.concept.name;
    g.append(name);
    g.addEventListener("click", () => showConcept(n.concept.id));
    g.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); showConcept(n.concept.id); }
    });
    box.append(g);
  }

  if (!list.length) {
    const none = svg("text", { y: 80, "text-anchor": "middle", class: "edge-label" });
    none.textContent = "No link to another concept";
    box.append(none);
  }
}

// --- The links, each with the passage(s) it was read from ---------------------------
function conceptLink(c) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "link";
  button.textContent = c.name;
  button.addEventListener("click", () => showConcept(c.id));
  return button;
}

function renderLinks(data) {
  const list = $("map-links");
  list.replaceChildren();
  if (data.message) {
    const li = document.createElement("li");
    li.className = "meta";
    li.textContent = data.message;
    list.append(li);
  }
  const self = data.concept;
  for (const link of data.links) {
    const li = document.createElement("li");
    const sentence = document.createElement("div");
    sentence.className = "link-sentence";
    const [from, to] = link.direction === "out" ? [null, link.other] : [link.other, null];
    const relation = document.createElement("span");
    relation.className = "relation";
    relation.textContent = ` ${relationWords(link.relation)} → `;
    const selfName = document.createElement("strong");
    selfName.textContent = self.name;
    sentence.append(from ? conceptLink(from) : selfName, relation, to ? conceptLink(to) : selfName);
    li.append(sentence);

    for (const source of link.sources) {
      const quote = document.createElement("blockquote");
      quote.textContent = source.quote;
      li.append(quote);
      if (source.document_missing) {
        const note = document.createElement("p");
        note.className = "meta";
        note.textContent = `${source.document} was deleted after the concept map was built.`;
        li.append(note);
      }
    }
    const chips = document.createElement("div");
    chips.className = "chips";
    const words = wordsOf(self, link.other);
    for (const c of link.chunks) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = `${c.source} · lines ${c.start_line}–${c.end_line}`;
      chip.addEventListener("click", () => openWordChunk(c, words));
      chips.append(chip);
    }
    li.append(chips);
    list.append(li);
  }
}

// --- The documents that mention the concept -----------------------------------------
function renderMentions(data) {
  const box = $("map-mentions");
  box.replaceChildren();
  const total = data.mentions.reduce((sum, m) => sum + m.occurrences, 0);
  $("map-docs").textContent = data.mentions.length
    ? `Open the documents (${data.mentions.length})` : "Open the documents";
  $("map-docs").disabled = !data.mentions.length;
  if (!data.mentions.length) {
    const p = document.createElement("p");
    p.className = "meta";
    p.textContent = `No document contains the name "${data.concept.name}" as such: the concept map got it from a passage `
      + "that writes it differently.";
    box.append(p);
    return;
  }
  const words = wordsOf(data.concept);
  for (const m of data.mentions) {
    const h = document.createElement("h4");
    h.textContent = `${m.document} — ${m.occurrences === 1 ? "1 mention" : `${m.occurrences} mentions`}, `
      + `in ${m.chunk_count === 1 ? "1 passage" : `${m.chunk_count} passages`}`;
    const ol = document.createElement("ol");
    ol.className = "retrieved";
    for (const c of m.chunks) ol.append(chunkListItem(c, words, `lines ${c.start_line}–${c.end_line}`));
    box.append(h, ol);
    if (m.chunk_count > m.chunks.length) {
      const more = document.createElement("p");
      more.className = "meta";
      more.textContent = `… and ${m.chunk_count - m.chunks.length} more passages.`;
      box.append(more);
    }
  }
  $("map-mentions-title").title = `${total} mention(s) in all`;
}

$("map-docs").addEventListener("click", () => {
  $("map-mentions-title").scrollIntoView({ behavior: "smooth", block: "start" });
});

// --- "Ask about this": go to the Ask tab with a question ready (the user can edit it) --
$("map-ask").addEventListener("click", () => {
  if (!map.current) return;
  $("question").value = questionAbout(map.current.concept);
  // The graph retriever is the one that follows the links of this concept: switch it on if it can be.
  const graphBox = document.querySelector('#retriever-list input[value="graph"]');
  if (graphBox && !graphBox.disabled) graphBox.checked = true;
  document.querySelector('.tab[data-tab="ask"]').click();
  $("question").focus();
});

// --- Loading: the first time the tab is opened, and again when the corpus changes -----
document.querySelector('.tab[data-tab="map"]').addEventListener("click", () => {
  if (map.corpus !== corpus()) loadConceptList();
});

$("corpus").addEventListener("change", () => {
  map.corpus = null;
  map.current = null;
  map.history = [];
  $("map-search").value = "";
  if (!$("tab-map").classList.contains("hidden")) loadConceptList();
});
