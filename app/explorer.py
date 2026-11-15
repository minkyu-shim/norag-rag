import time
from collections import Counter
from contextlib import contextmanager
from app import documents, graph
from app.chunking import coverage
from app.graph_retrieval import load_graph, alias_pattern, chunks_of, HUB_DEGREE, MIN_COVERAGE

# V4 — Part B: the CONCEPT MAP explorer. The V3 graph, made something a user can explore.
#
#   list of concepts  ->  click one  ->  its neighbours, the relation of each link,
#                                        the quote and the chunks each link was extracted from,
#                                        and the documents that mention the concept
#
# Decisions (see REPORT.md):
#   - Read-only, NO LLM call: the graph is already built (scripts/build_graph.py). A click costs
#     a few milliseconds, the cost of the feature is the cost of the build (44 LLM calls, once).
#   - The neighbourhood of ONE concept (1 hop), not the whole graph: the user explores step by step,
#     and the drawing stays readable (on Kestrel no concept has more than 4 distinct links).
#   - A link points to its chunks of the CURRENT chunk setting: the edge stores its quote's character
#     span, mapped onto the chunks with coverage(), same rule as the graph retriever and the evaluation.
#   - "Documents that mention it" = a plain text search of the concept's names, with the same matching
#     rule as the graph retriever (graph_retrieval.alias_pattern). It also finds documents the graph
#     never extracted a triple from.
#   - The response carries the chunk TEXTS (the UI opens them); the timings of the steps only carry ids.

MAX_MENTION_CHUNKS = 10     # per document: enough to read, short enough to scroll


class ExplorerError(Exception):
    status_code = 400


class GraphMissingError(ExplorerError):
    """No concept graph for this corpus: the user must build it (minutes, LLM calls)."""


class ConceptNotFoundError(ExplorerError):
    status_code = 404


class Steps:
    # Timing of each step of one request: shown to the user ("12 ms · 0 LLM calls").
    # TODO(V4 Part A): each step becomes a tracing.span(...) once app/tracing.py exists.
    def __init__(self):
        self.started = time.perf_counter()
        self.steps = []

    @contextmanager
    def step(self, name, **attributes):
        start = time.perf_counter()
        record = {"name": name, **attributes}
        try:
            yield record            # the caller adds what it found: record["edges"] = 4 (ids and counts only)
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000, 1)
            self.steps.append(record)

    def cost(self):
        return {"llm_calls": 0, "ms": round((time.perf_counter() - self.started) * 1000, 1), "steps": self.steps}


def require_graph(corpus):
    g = load_graph(corpus)
    if g is None:
        raise GraphMissingError(
            f"There is no concept map for {corpus!r} yet. It is built once, from the terminal: "
            f"python scripts/build_graph.py {corpus}  (one LLM call per chunk: 44 calls, ~7 minutes for Kestrel).")
    return g


def distinct_links(g, node_id):
    # The same triple can be extracted twice, from two overlapping chunks (Payments Core --lead--> Jonas Berg
    # is in service-catalog.md::3 AND ::4): the graph then has two edges for ONE link. The user sees one.
    return ({("out", e["relation"], target) for _, target, e in g.out_edges(node_id, data=True)}
            | {("in", e["relation"], source) for source, _, e in g.in_edges(node_id, data=True)})


def concept_info(g, node_id):
    node = g.nodes[node_id]
    return {"id": node_id, "name": node["name"], "type": node["type"], "aliases": node["aliases"],
            "links": len(distinct_links(g, node_id)), "hub": g.degree(node_id) > HUB_DEGREE}


# --- The list of concepts (search box + list grouped by type) ----------------------------
def concepts(corpus, query=""):
    steps = Steps()
    with steps.step("graph.load") as s:
        g = require_graph(corpus)
        s["nodes"] = g.number_of_nodes()
    with steps.step("concepts.search", query=query) as s:
        q = query.strip().casefold()
        found = [concept_info(g, n) for n in g.nodes
                 if not q or q in g.nodes[n]["type"].casefold()
                 or any(q in alias.casefold() for alias in g.nodes[n]["aliases"])]
        found.sort(key=lambda c: (c["type"], c["name"].casefold()))
        s["found"] = len(found)
    built = graph.load(corpus)
    return {
        "corpus": corpus,
        "built": built["built"],
        "total": g.number_of_nodes(),
        "relations": g.number_of_edges(),
        "types": dict(Counter(c["type"] for c in found)),
        "concepts": found,
        "cost": steps.cost(),
    }


# --- One concept: its neighbours, and the chunks behind each link --------------------------
def chunk_dict(chunk_id, document, chunk):
    return {"id": chunk_id, "source": document, "start": chunk["start"], "end": chunk["end"],
            "start_line": chunk["start_line"], "end_line": chunk["end_line"], "text": chunk["text"]}


class _Span:
    def __init__(self, chunk):
        self.start, self.end = chunk["start"], chunk["end"]


def document_chunks(corpus, document, settings, cache):
    # The chunks of one document for the current setting, or None if the document was deleted since the build.
    if document not in cache:
        try:
            cache[document] = chunks_of(corpus, document, settings)
        except FileNotFoundError:
            cache[document] = None
    return cache[document]


def concept(corpus, concept_id, settings):
    steps = Steps()
    with steps.step("graph.load"):
        g = require_graph(corpus)
    if concept_id not in g:
        raise ConceptNotFoundError(f"No concept {concept_id!r} in the concept map of {corpus!r}.")

    with steps.step("graph.neighbours", concept=concept_id) as s:
        edges = [("out", source, target, edge) for source, target, edge in g.out_edges(concept_id, data=True)]
        edges += [("in", source, target, edge) for source, target, edge in g.in_edges(concept_id, data=True)]
        s["edges"] = len(edges)

    cache = {}
    links = {}          # (direction, relation, other id) -> link: duplicate edges become one link
    with steps.step("chunks.map", chunk_size=settings.chunk_size, overlap_percent=settings.overlap_percent) as s:
        for direction, source, target, edge in edges:
            other = target if direction == "out" else source
            link = links.setdefault((direction, edge["relation"], other), {
                "direction": direction,             # out: concept --relation--> other; in: other --relation--> concept
                "relation": edge["relation"],
                "other": concept_info(g, other),
                "sources": [],                      # one per edge: where the build read this link
                "chunks": [],
            })
            chunks = document_chunks(corpus, edge["document"], settings, cache)
            link["sources"].append({"quote": edge["quote"], "document": edge["document"],
                                    "document_missing": chunks is None})
            span = tuple(edge["span"])
            for cid, c in chunks or []:
                if coverage(_Span(c), span) >= MIN_COVERAGE and cid not in {x["id"] for x in link["chunks"]}:
                    link["chunks"].append(chunk_dict(cid, edge["document"], c))
        links = sorted(links.values(), key=lambda l: (l["relation"], l["direction"], l["other"]["name"].casefold()))
        s["links"] = len(links)
        s["chunk_ids"] = sorted({c["id"] for l in links for c in l["chunks"]})

    with steps.step("documents.mentions") as s:
        mentions = mentioning_documents(corpus, g.nodes[concept_id], settings, cache)
        s["documents"] = [m["document"] for m in mentions]

    info = concept_info(g, concept_id)
    message = None
    if not links:
        message = (f"{info['name']} has no link in the concept map: the build found it, but no relation "
                   f"between it and another concept.")
    return {"corpus": corpus, "concept": info, "links": links, "mentions": mentions,
            "message": message, "cost": steps.cost()}


def mentioning_documents(corpus, node, settings, cache):
    # The documents whose text contains one of the concept's names, with the chunks that do.
    patterns = [p for p in map(alias_pattern, node["aliases"]) if p is not None]
    result = []
    for path in documents.list_document_paths(corpus):
        text = documents.read_document(corpus, path.name)
        count = sum(len(p.findall(text)) for p in patterns)
        if not count:
            continue
        chunks = document_chunks(corpus, path.name, settings, cache) or []
        hits = [chunk_dict(cid, path.name, c) for cid, c in chunks if any(p.search(c["text"]) for p in patterns)]
        result.append({"document": path.name, "occurrences": count, "chunk_count": len(hits),
                       "chunks": hits[:MAX_MENTION_CHUNKS]})
    result.sort(key=lambda m: -m["occurrences"])
    return result
