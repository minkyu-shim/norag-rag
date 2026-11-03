import argparse
import sys
from pathlib import Path

# Make "import app..." work when running "python eval/run_eval.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import evaluation, llm  # noqa: E402
from app.config import CORPORA, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_PERCENT, DEFAULT_TOP_K  # noqa: E402
from app.schemas import Settings  # noqa: E402

# Retrieval evaluation from the terminal. The measuring code is in app/evaluation.py
# (shared with the Eval tab of the UI); this file only reads the options and prints the table.
#
#   python eval/run_eval.py kestrel                                # chunk 800, overlap 10%, k = 1,3,5,10
#   python eval/run_eval.py crime --chunk-size 500 --overlap 20    # another setting (indexed if needed)
#   python eval/run_eval.py kestrel --llm                          # also generate answers (costs API calls)
#
# Results: eval/results/<corpus>_c<size>_o<overlap>[_llm]_<date>.json


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation (hit@k, recall@k) on a corpus' question set")
    parser.add_argument("corpus", choices=list(CORPORA))
    # Same defaults as the app (app/config.py), so a plain run measures what the app uses.
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_PERCENT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="chunks given to the LLM (with --llm)")
    parser.add_argument("--k", default="1,3,5,10", help="comma-separated list of k values")
    parser.add_argument("--llm", action="store_true", help="also generate the answers with the LLM")
    args = parser.parse_args()

    # Without a key, every LLM row would be a provider_error: a silently wrong run. Better to stop right away.
    if args.llm and llm.client is None:
        raise SystemExit("--llm needs GOOGLE_API_KEY: create .env from .env.example, or run without --llm.")

    ks = sorted(int(k) for k in args.k.split(","))
    settings = Settings(chunk_size=args.chunk_size, overlap_percent=args.overlap, top_k=args.top_k)
    result = evaluation.evaluate(args.corpus, settings, ks, with_llm=args.llm)
    print_report(result, ks)
    path = evaluation.save(result)
    print(f"\nSaved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


def print_report(result, ks):
    s = result["settings"]
    summary = result["summary"]
    print(f"\n{result['corpus']} — chunk {s['chunk_size']}, overlap {s['overlap_percent']}% — "
          f"relevant = chunk covers >= {s['min_coverage']:.0%} of an evidence quote\n")
    header = (f"{'type':17} {'n':>3} " + " ".join(f"{'hit@' + str(k):>7}" for k in ks)
              + "  " + " ".join(f"{'rec@' + str(k):>7}" for k in ks))
    print(header)
    print("-" * len(header))
    for name, line in summary.items():
        if name == "top1_distance":
            continue
        print(f"{name:17} {line['n']:>3} " + " ".join(f"{line[f'hit@{k}']:>7.2f}" for k in ks)
              + "  " + " ".join(f"{line[f'recall@{k}']:>7.2f}" for k in ks))

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
