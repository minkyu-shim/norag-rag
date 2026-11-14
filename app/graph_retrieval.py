import re
import networkx as nx
from app import documents, graph
from app.chunking import chunk_text, coverage
from app.schemas import ChunkRef

# V3 — The concept graph, part 2: the graph as a RETRIEVER (answering phase, no LLM call).
#
# Strategy: LOCAL WALK, then RRF.
#   1. entities of the question: the graph nodes whose name appears in the question (string match)
#   2. walk: follow the edges (both directions) up to MAX_HOPS from those entities
#   3. every edge followed points to its quote -> the chunks that contain the quote
#   4. rank the chunks: smallest hop first (closer to the question), then the chunks backing more edges
# The ranked chunk list goes into RRF like the dense and BM25 lists (see rag.py).
#
# Alternatives we did not pick (see REPORT.md):
#   - expansion: run dense+BM25 first, add the chunks of the entities found in their top k.
#     Simpler, but hop 1 must be found by dense/BM25 first: the graph cannot rescue a miss.
#   - path as context: give the triples themselves to the LLM (the teams-of-3 option).
#
# Storage: the graph JSON (data/graph/<corpus>.json) is loaded into a NetworkX MultiDiGraph
# (several relations between the same two nodes are allowed). ~100 nodes: memory is enough, no Neo4j.

# q08 needs 3 hops: KST-5031 -raised_by-> postroom -owned_by-> Messaging -on_call_channel-> #oncall-messaging
MAX_HOPS = 3
# A HUB is a node with many edges (e.g. "Platform", owner of several services and the fallback on-call).
# Walking through it would reach most of the graph. A hub is reached, but not walked through
# (unless the question names it: then it is a seed, and its own edges are what the question is about).
HUB_DEGREE = 12
# An edge's quote "is in" a chunk when the chunk covers at least this share of it (same rule as the evaluation).
MIN_COVERAGE = 0.5

_loaded = {}        # corpus -> (graph file modification time, MultiDiGraph): reloaded when the file changes


def load_graph(corpus):
    path = graph.graph_file(corpus)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if corpus not in _loaded or _loaded[corpus][0] != mtime:
        data = graph.load(corpus)
        g = nx.MultiDiGraph()
        for node in data["nodes"]:
            g.add_node(node["id"], **node)
        for edge in data["edges"]:
            g.add_edge(edge["source"], edge["target"], **edge)
        _loaded[corpus] = (mtime, g)
    return _loaded[corpus][1]


def is_available(corpus):
    return graph.graph_file(corpus).exists()


# --- 1. Entities of the question ----------------------------------------------------
# No LLM: we look for every name (and alias) of every node in the question.
#   - identifiers and names of several words ("KST-5031", "Data Integrations"): case-insensitive
#   - names of ONE plain word must match with their case: the plan "Team" must not match "which team",
#     the plan "Free" must not match "free a seat". (Services are written in lowercase, like the docs.)
# Known limit: a question that names nothing of the graph ("I lost my phone") gets no seed at all.
def find_entities(question, g):
    seeds = []
    for node_id, node in g.nodes(data=True):
        for alias in node["aliases"]:
            name = alias.strip("*`# ")
            if len(name) < 3:
                continue
            one_plain_word = re.fullmatch(r"[A-Za-zÀ-ÿ]+", name) is not None
            pattern = rf"(?<![\w-]){re.escape(name)}(?![\w-])"
            if re.search(pattern, question, flags=0 if one_plain_word else re.IGNORECASE):
                seeds.append(node_id)
                break
    return seeds


# --- 2. Walk -------------------------------------------------------------------------
def walk(g, seeds):
    """Breadth-first walk from the seeds. Returns the edges followed, each with its hop (1 = touches a seed)."""
    visited = set(seeds)
    frontier = list(seeds)
    followed = []
    seen_edges = set()
    for hop in range(1, MAX_HOPS + 1):
        next_frontier = []
        for node in frontier:
            if node not in seeds and g.degree(node) > HUB_DEGREE:
                continue                    # a hub: reached, but not walked through
            # Both directions: "which team owns X" goes forward, "which services does team Y own" goes back.
            for source, target, key, edge in [*g.out_edges(node, keys=True, data=True),
                                              *g.in_edges(node, keys=True, data=True)]:
                if (source, target, key) in seen_edges:
                    continue
                seen_edges.add((source, target, key))
                followed.append({**edge, "hop": hop})
                neighbour = target if source == node else source
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier
    return followed


# --- 3 + 4. Edges -> chunks of the CURRENT setting, ranked ----------------------------
def chunks_of(corpus, name, settings):
    # The same chunks and ids as the indexing (documents.index_document): chunking is deterministic.
    text = documents.read_document(corpus, name)
    return [(f"{name}::{i}", c) for i, c in enumerate(chunk_text(text, settings.chunk_size, settings.overlap_percent))]


class _Span:
    # coverage() reads .start / .end: a tiny adapter for a chunk dict.
    def __init__(self, chunk):
        self.start, self.end = chunk["start"], chunk["end"]


def explain(corpus, question, sources, settings):
    """The whole walk, for the retriever AND for the UI: {"entities", "edges", "chunk_ids" (ranked)}."""
    g = load_graph(corpus)
    if g is None:
        return {"entities": [], "edges": [], "chunk_ids": []}
    seeds = find_entities(question, g)
    followed = walk(g, seeds)

    chunk_cache = {}
    best = {}           # chunk id -> (smallest hop, number of edges, chunk dict, document)
    edges_out = []
    for edge in followed:
        if edge["document"] not in sources:
            continue                # the user did not select this document
        if edge["document"] not in chunk_cache:
            chunk_cache[edge["document"]] = chunks_of(corpus, edge["document"], settings)
        span = tuple(edge["span"])
        chunk_ids = []
        for chunk_id, chunk in chunk_cache[edge["document"]]:
            if coverage(_Span(chunk), span) >= MIN_COVERAGE:
                chunk_ids.append(chunk_id)
                hop, count, _, _ = best.get(chunk_id, (edge["hop"], 0, None, None))
                best[chunk_id] = (min(hop, edge["hop"]), count + 1, chunk, edge["document"])
        edges_out.append({
            "source": g.nodes[edge["source"]]["name"], "relation": edge["relation"],
            "target": g.nodes[edge["target"]]["name"], "hop": edge["hop"],
            "quote": edge["quote"], "document": edge["document"], "chunk_ids": chunk_ids,
        })

    # Smallest hop first; on a tie, the chunk that backs more edges first.
    ranked = sorted(best, key=lambda cid: (best[cid][0], -best[cid][1]))
    return {
        "entities": [g.nodes[s]["name"] for s in seeds],
        "edges": edges_out,
        "chunk_ids": ranked,
        "_chunks": {cid: (best[cid][2], best[cid][3]) for cid in ranked},
    }


def retrieve(corpus, question, sources, settings, n):
    # Same shape as the other retrievers (see rag.RETRIEVERS): the n best chunks, best first.
    result = explain(corpus, question, sources, settings)
    chunks = []
    for i, chunk_id in enumerate(result["chunk_ids"][:n]):
        chunk, document = result["_chunks"][chunk_id]
        chunks.append(ChunkRef(
            label=f"C{i + 1}", id=chunk_id, source=document, start=chunk["start"], end=chunk["end"],
            start_line=chunk["start_line"], end_line=chunk["end_line"], text=chunk["text"],
        ))
    return chunks
