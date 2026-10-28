import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Make "import app..." work when running "python eval/run_eval.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import documents, rag, llm  # noqa: E402
from app.schemas import Settings  # noqa: E402

# Retrieval baseline: how often is the RIGHT PASSAGE in the top-k?
#
#   python eval/run_eval.py                                  # chunk 800, overlap 10%, k = 1,3,5
#   python eval/run_eval.py --chunk-size 500 --overlap 20    # another setting (indexed if needed)
#   python eval/run_eval.py --llm                            # also ask the LLM (costs API calls)
#
# How "the right passage" is defined, so that the numbers stay comparable when chunking changes:
# each question stores a QUOTE of its evidence, not a chunk id. We find the quote in the file to get
# its character span [start, end). A retrieved chunk is a HIT when it covers at least
# MIN_COVERAGE of that span. Chunk ids change with every setting, character offsets do not.
#
# Retrieval only (default) = no LLM call: free, fast and deterministic, so you can re-run it at will.

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MIN_COVERAGE = 0.5
LLM_DELAY_SECONDS = 4.5   # stay under the free-tier rate limit (~15 requests/minute)


def find_span(text, quote):
    # The book is hard-wrapped (a newline every ~70 characters), so a quote may cross line breaks.
    # We match it word by word, with any whitespace (spaces, newlines) between the words.
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, text)
    if match is None:
        raise SystemExit(f"Evidence quote not found in the source, fix questions.json: {quote!r}")
    return match.start(), match.end()


def coverage(chunk, span):
    overlap = min(chunk.end, span[1]) - max(chunk.start, span[0])
    return max(overlap, 0) / (span[1] - span[0])


def first_hit_rank(retrieved, source, span):
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source == source and coverage(chunk, span) >= MIN_COVERAGE:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser(description="Retrieval baseline (hit rate @ k)")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--k", default="1,3,5", help="comma-separated list of k values")
    parser.add_argument("--llm", action="store_true", help="also run the full pipeline with the LLM")
    args = parser.parse_args()

    # Without a key, every LLM row would be a provider_error and the LLM numbers would read 0/n:
    # a silently wrong measure. Better to stop right away.
    if args.llm and llm.client is None:
        raise SystemExit("--llm needs GOOGLE_API_KEY: create .env from .env.example, or run without --llm.")

    ks = sorted(int(k) for k in args.k.split(","))
    max_k = ks[-1]
    settings = Settings(chunk_size=args.chunk_size, overlap_percent=args.overlap, top_k=max_k)
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    # Make sure every source document is indexed with THIS setting (same code as the app).
    documents.seed_corpus()
    sources = sorted({q["source"] for q in questions if q["source"]})
    for source in sources:
        if not documents.is_indexed(source, settings):
            documents.index_document(source, settings)
    texts = {source: documents.read_document(source) for source in sources}

    rows = []
    for q in questions:
        # Unanswerable questions are searched in every source document.
        search_in = [q["source"]] if q["source"] else sources
        retrieved = rag.retrieve(q["question"], search_in, settings)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "top1_distance": round(retrieved[0].distance, 4) if retrieved else None,
            "retrieved": [{"id": c.id, "lines": f"{c.start_line}-{c.end_line}", "distance": round(c.distance, 4)}
                          for c in retrieved],
        }
        if q["evidence_quote"]:
            span = find_span(texts[q["source"]], q["evidence_quote"])
            row["evidence_span"] = span
            row["rank"] = first_hit_rank(retrieved, q["source"], span)

        if args.llm:
            response, _ = rag.answer(q["question"], search_in, settings)
            row["llm_status"] = response.status
            row["llm_answer"] = response.answer
            if q["evidence_quote"]:
                # The answer is "grounded" if one of its citations covers the evidence.
                row["llm_cited_evidence"] = first_hit_rank(response.citations, q["source"], span) is not None
            time.sleep(LLM_DELAY_SECONDS)

        rows.append(row)

    # --- Report ---------------------------------------------------------------
    answerable = [r for r in rows if "rank" in r]
    groups = {"ALL answerable": answerable}
    for t in ["easy", "hard", "needle"]:
        groups[t] = [r for r in answerable if r["type"] == t]

    print(f"\nRetrieval baseline — chunk {args.chunk_size}, overlap {args.overlap}%, "
          f"hit = chunk covers >= {MIN_COVERAGE:.0%} of the evidence\n")
    header = f"{'group':16} {'n':>3} " + " ".join(f"{'hit@' + str(k):>7}" for k in ks) + f" {'MRR@' + str(max_k):>7}"
    print(header)
    print("-" * len(header))
    summary = {}
    for name, group in groups.items():
        if not group:
            continue
        hits = {k: sum(1 for r in group if r["rank"] and r["rank"] <= k) / len(group) for k in ks}
        mrr = sum(1 / r["rank"] for r in group if r["rank"]) / len(group)
        summary[name] = {"n": len(group), **{f"hit@{k}": round(v, 3) for k, v in hits.items()}, "mrr": round(mrr, 3)}
        print(f"{name:16} {len(group):>3} " + " ".join(f"{hits[k]:>7.2f}" for k in ks) + f" {mrr:>7.2f}")

    # Top-1 distance: is "nothing relevant" visible in the scores? (Material for a V2 threshold.)
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    def mean(values):
        return sum(values) / len(values) if values else float("nan")
    print(f"\nMean top-1 distance: answerable {mean([r['top1_distance'] for r in answerable]):.3f}, "
          f"unanswerable {mean([r['top1_distance'] for r in unanswerable]):.3f} (lower = closer)")

    misses = [r for r in answerable if not r["rank"]]
    if misses:
        print("\nMissed (evidence not in top-{}):".format(max_k))
        for r in misses:
            print(f"  {r['id']}  {r['question']}")

    if args.llm:
        # Questions where the provider failed (429, timeout...) say nothing about the pipeline:
        # they are counted apart, not as wrong answers.
        errors = [r for r in rows if r["llm_status"] == "provider_error"]
        unanswerable_ok = [r for r in unanswerable if r["llm_status"] != "provider_error"]
        answerable_ok = [r for r in answerable if r["llm_status"] != "provider_error"]
        not_found = sum(1 for r in unanswerable_ok if r["llm_status"] == "not_found")
        grounded = sum(1 for r in answerable_ok if r.get("llm_cited_evidence"))
        print(f"\nLLM: unanswerable -> not_found {not_found}/{len(unanswerable_ok)}; "
              f"answerable with a citation covering the evidence {grounded}/{len(answerable_ok)}; "
              f"provider errors (excluded) {len(errors)}/{len(rows)}")
        for r in rows:
            print(f"  {r['id']}  {r['llm_status']:15} {(r['llm_answer'] or '')[:90]!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_c{args.chunk_size}_o{args.overlap}.json"
    out.write_text(json.dumps({
        "settings": {"chunk_size": args.chunk_size, "overlap_percent": args.overlap, "ks": ks,
                     "min_coverage": MIN_COVERAGE, "llm": args.llm},
        "summary": summary,
        "questions": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")


if __name__ == "__main__":
    main()
