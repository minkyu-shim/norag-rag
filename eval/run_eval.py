import argparse
import sys
from pathlib import Path
from typing import get_args

# Make "import app..." work when running "python eval/run_eval.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import evaluation, llm  # noqa: E402
from app.config import CORPORA, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_PERCENT, DEFAULT_TOP_K  # noqa: E402
from app.schemas import Settings, RetrieverName, DEFAULT_RETRIEVERS  # noqa: E402

# Retrieval evaluation from the terminal. The measuring code is in app/evaluation.py
# (shared with the Eval tab of the UI); this file only reads the options and prints the table.
#
#   python eval/run_eval.py kestrel                                # chunk 800, overlap 10%, k = 1,3,5,10
#   python eval/run_eval.py crime --chunk-size 500 --overlap 20    # another setting (indexed if needed)
#   python eval/run_eval.py kestrel --llm                          # also generate answers (costs API calls)
#   python eval/run_eval.py kestrel --retrievers dense             # V3: choose the retrievers (fused with RRF)
#   python eval/run_eval.py kestrel --retrievers dense,bm25,graph
#
# Results: eval/results/<corpus>_<retrievers>_c<size>_o<overlap>[_llm]_<date>.json


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation (hit@k, recall@k) on a corpus' question set")
    parser.add_argument("corpus", choices=list(CORPORA))
    # Same defaults as the app (app/config.py), so a plain run measures what the app uses.
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_PERCENT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="chunks given to the LLM (with --llm)")
    parser.add_argument("--k", default="1,3,5,10", help="comma-separated list of k values")
    parser.add_argument("--llm", action="store_true", help="also generate the answers with the LLM")
    parser.add_argument("--retrievers", default=",".join(DEFAULT_RETRIEVERS),
                        help=f"comma-separated, fused with RRF. Known: {', '.join(get_args(RetrieverName))}")
    args = parser.parse_args()

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    unknown = [r for r in retrievers if r not in get_args(RetrieverName)]
    if unknown or not retrievers:
        parser.error(f"unknown retriever(s): {', '.join(unknown) or '(none given)'}")

    # Without a key, every LLM row would be a provider_error: a silently wrong run. Better to stop right away.
    if args.llm and llm.client is None:
        raise SystemExit("--llm needs GOOGLE_API_KEY: create .env from .env.example, or run without --llm.")

    ks = sorted(int(k) for k in args.k.split(","))
    settings = Settings(chunk_size=args.chunk_size, overlap_percent=args.overlap, top_k=args.top_k)
    result = evaluation.evaluate(args.corpus, settings, ks, retrievers=retrievers, with_llm=args.llm)
    print_report(result, ks)
    path = evaluation.save(result)
    print(f"\nSaved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def print_report(result, ks):
    s = result["settings"]
    summary = result["summary"]
    print(f"\n{result['corpus']} — retrievers {evaluation.config_label(s['retrievers'])} "
          f"(RRF k={s['rrf_k']}, {s['rrf_candidates']} candidates each) — chunk {s['chunk_size']}, "
          f"overlap {s['overlap_percent']}% — relevant = chunk covers >= {s['min_coverage']:.0%} of an evidence quote\n")
    header = (f"{'type':17} {'n':>3} " + " ".join(f"{'hit@' + str(k):>7}" for k in ks)
              + "  " + " ".join(f"{'rec@' + str(k):>7}" for k in ks))
    print(header)
    print("-" * len(header))
    groups = [name for name in summary if "n" in summary[name]]     # ALL + the question types
    for name in groups:
        line = summary[name]
        print(f"{name:17} {line['n']:>3} " + " ".join(f"{line[f'hit@{k}']:>7.2f}" for k in ks)
              + "  " + " ".join(f"{line[f'recall@{k}']:>7.2f}" for k in ks))

    # Each retriever's own list, before the fusion. "lost" = it had the evidence in its top k, the fused top k did not.
    if len(s["retrievers"]) > 1:
        print(f"\nBefore fusion (each retriever's own list): hit@k, and questions lost by the fusion")
        for retriever, per_type in summary["per_retriever"].items():
            line = per_type["ALL"]
            lost = sorted({qid for k in ks for qid in line[f"lost@{k}"]})
            print(f"  {retriever:6} " + " ".join(f"hit@{k}={line[f'hit@{k}']:.2f}" for k in ks)
                  + (f"   lost: {', '.join(lost)}" if lost else ""))

    c = summary["cost_per_question"]
    print(f"\nCost per question: retrieval {c['retrieval_ms']} ms"
          + (f", {c['llm_calls']} LLM calls, {c['prompt_tokens']:.0f} + {c['completion_tokens']:.0f} tokens, "
             f"LLM {c['llm_ms']} ms" if s["with_llm"] else ""))
    if "unanswerable" in summary:
        u = summary["unanswerable"]
        print(f"Unanswerable: {u['abstained']}/{u['n']} abstained, {u['hallucinated']} hallucinated, {u['errors']} errors")

    d = summary["top1_distance"]
    print(f"\nMean top-1 distance: answerable {d['answerable']}, unanswerable {d['unanswerable']} (lower = closer)")

    rows = result["questions"]
    misses = [r for r in rows if r["answerable"] and not r["rank"]]
    if misses:
        print(f"\nMissed (no relevant chunk in the top {ks[-1]}):")
        for r in misses:
            print(f"  {r['id']}  {r['type']:17} {r['question']}")

    if s["with_llm"]:
        print("\nLLM answers (grade them by hand in the results file):")
        for r in rows:
            print(f"  {r['id']}  {r['llm']['status']:15} {(r['llm']['answer'] or '')[:90]!r}")


if __name__ == "__main__":
    main()
