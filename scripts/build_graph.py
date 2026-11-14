import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make "import app..." work when running "python scripts/build_graph.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import graph, llm  # noqa: E402
from app.config import CORPORA, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_PERCENT, DEFAULT_TOP_K  # noqa: E402
from app.evaluation import RESULTS_DIR  # noqa: E402
from app.schemas import Settings  # noqa: E402

# V3 — Build the concept graph of a corpus (one LLM call per chunk; see app/graph.py).
#
#   python scripts/build_graph.py kestrel                                  # chunk 800, overlap 10%
#   python scripts/build_graph.py kestrel --chunk-size 500 --overlap 20
#
# The graph goes to data/graph/<corpus>.json (used by the app and the evaluation, for any chunk setting).
# The build report (cost, dropped triples, merges) AND a copy of the graph go to
#   eval/results/graph_<corpus>_c<size>_o<overlap>_<date>.json   (committed: the graph is a deliverable)
# Extractions are cached per chunk (data/graph/cache/): a build stopped by 503 errors resumes for free,
# and a second build costs 0 LLM calls.


def main():
    parser = argparse.ArgumentParser(description="Build the concept graph of a corpus with the LLM")
    parser.add_argument("corpus", choices=list(CORPORA))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_PERCENT)
    args = parser.parse_args()
    if llm.client is None:
        raise SystemExit("Building the graph needs GOOGLE_API_KEY: create .env from .env.example.")

    settings = Settings(chunk_size=args.chunk_size, overlap_percent=args.overlap, top_k=DEFAULT_TOP_K)
    report, built = graph.build(args.corpus, settings)

    c, t, e = report["cost"], report["triples"], report["entities"]
    print(f"\nGraph of {args.corpus} (chunk {args.chunk_size}, overlap {args.overlap}%):")
    print(f"  cost:      {c['llm_calls']} LLM calls ({c['cached_chunks']} chunks from cache), "
          f"{c['prompt_tokens']} + {c['completion_tokens']} tokens, {c['seconds']} s")
    if c["failed_chunks"]:
        print(f"  FAILED:    {len(c['failed_chunks'])} chunks, run again to retry them: {', '.join(c['failed_chunks'])}")
    print(f"  triples:   {t['raw']} extracted, {t['valid']} valid, dropped {t['dropped'] or 0}")
    print(f"  entities:  {e['surface_names']} names -> {e['nodes_after_merge']} nodes, {len(e['merges'])} spelling merges, "
          f"{len(e['cross_chunk_merges'])} entities linking several chunks, "
          f"type conflicts: {e['type_conflicts'] or 'none'}")
    print(f"  graph:     {report['graph_stats']}")
    print(f"  saved:     {graph.graph_file(args.corpus)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"graph_{args.corpus}_c{args.chunk_size}_o{args.overlap}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({**report, "graph": built}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  report:    {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
