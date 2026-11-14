import argparse
import json
import random
import sys
from pathlib import Path

# Make "import app..." work when running "python eval/sample_triples.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import documents, graph  # noqa: E402
from app.evaluation import EVAL_DIR  # noqa: E402

# V3 — Extraction quality of the concept graph: pick 10 random triples, to be graded BY HAND.
#
#   python eval/sample_triples.py kestrel            # writes eval/graph_triples_graded_kestrel.json
#
# For each triple the file holds the triple, its quote and the whole passage around the quote
# (the source chunk), so the grader can judge it without opening the file. Fill "grade" with:
#   correct      the text states this fact, and the triple says it right
#   wrong        the text says something else (wrong entity, wrong type, wrong direction...)
#   unsupported  the text does not state this fact (the LLM inferred or invented it)
# The seed is fixed: the same graph always gives the same 10 triples (re-running does not cherry-pick).
SEED = 42
SAMPLE_SIZE = 10
CONTEXT_CHARS = 400     # characters shown before and after the quote


def main():
    parser = argparse.ArgumentParser(description="Sample graph triples for hand grading")
    parser.add_argument("corpus")
    parser.add_argument("--force", action="store_true", help="overwrite the file (loses the grades)")
    args = parser.parse_args()

    out = EVAL_DIR / f"graph_triples_graded_{args.corpus}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.name} already exists (it may hold your grades). Use --force to overwrite it.")
    data = graph.load(args.corpus)
    if data is None:
        raise SystemExit(f"No graph for {args.corpus}: run  python scripts/build_graph.py {args.corpus}")

    names = {n["id"]: n["name"] for n in data["nodes"]}
    types = {n["id"]: n["type"] for n in data["nodes"]}
    sample = random.Random(SEED).sample(data["edges"], min(SAMPLE_SIZE, len(data["edges"])))

    rows = []
    for i, e in enumerate(sample, start=1):
        text = documents.read_document(args.corpus, e["document"])
        start, end = e["span"]
        rows.append({
            "n": i,
            "triple": f"{names[e['source']]} ({types[e['source']]}) --{e['relation']}--> "
                      f"{names[e['target']]} ({types[e['target']]})",
            "quote": e["quote"],
            "document": e["document"],
            "chunk_id": e["chunk_id"],
            "context": text[max(0, start - CONTEXT_CHARS):end + CONTEXT_CHARS],
            "grade": None,          # correct / wrong / unsupported (BY HAND)
            "note": None,
        })

    result = {"corpus": args.corpus, "graph_built": data["built"], "seed": SEED,
              "edges_in_graph": len(data["edges"]), "triples": rows}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} triples to {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    for r in rows:
        print(f"  {r['n']:2}. {r['triple']}")


if __name__ == "__main__":
    main()
