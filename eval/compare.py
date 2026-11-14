import argparse
import json
from pathlib import Path

# V3 — Compare evaluation runs: one markdown table per (corpus, metric), configurations × question types.
# Reads the result files written by eval/run_eval.py and prints tables ready to paste into REPORT.md.
#
#   python eval/compare.py eval/results/kestrel_*_c800_o10_2026110[9]*.json
#   python eval/compare.py eval/results/*_c800_o10_*.json --metrics hit@5,recall@5,hit@10
#
# When several files have the same corpus + configuration + chunk setting, the LATEST one is used
# (by its "date" field), so re-running a configuration simply replaces it in the table.
# Files from before V3 have no "retrievers" setting: they were dense only.


def config_name(settings):
    name = "+".join(settings.get("retrievers", ["dense"]))
    return f"{name} (c{settings['chunk_size']}/o{settings['overlap_percent']})"


def load_latest(paths):
    latest = {}
    for path in paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        key = (result["corpus"], config_name(result["settings"]))
        if key not in latest or result["date"] > latest[key]["date"]:
            latest[key] = result
    return latest


def main():
    parser = argparse.ArgumentParser(description="Markdown tables: configurations × question types")
    parser.add_argument("files", nargs="+", help="result files from eval/run_eval.py")
    parser.add_argument("--metrics", default="hit@5,recall@5", help="comma-separated, e.g. hit@1,hit@5,recall@10")
    args = parser.parse_args()

    latest = load_latest(args.files)
    for corpus in sorted({c for c, _ in latest}):
        runs = {name: r for (c, name), r in sorted(latest.items()) if c == corpus}
        # The question types, in the order of the first run ("ALL" first). Summary entries without hit@k
        # (per_retriever, cost, unanswerable...) are not question types.
        first_summary = next(iter(runs.values()))["summary"]
        types = [t for t in first_summary if any(key.startswith("hit@") for key in first_summary[t])]

        for metric in args.metrics.split(","):
            first = next(iter(runs.values()))["summary"]
            print(f"\n**{corpus} — {metric}**\n")
            print("| configuration | " + " | ".join(f"{t} (n={first[t]['n']})" for t in types) + " |")
            print("|---|" + "---|" * len(types))
            for name, result in runs.items():
                summary = result["summary"]
                cells = [f"{summary[t][metric]:.2f}" if t in summary else "–" for t in types]
                print(f"| {name} | " + " | ".join(cells) + " |")

        # Cost per question, one line per configuration.
        print(f"\n**{corpus} — cost per question**\n")
        print("| configuration | retrieval ms | LLM calls | prompt tokens | completion tokens | LLM ms |")
        print("|---|---|---|---|---|---|")
        for name, result in runs.items():
            c = result["summary"].get("cost_per_question")
            if c is None:
                print(f"| {name} | – | – | – | – | – |")
                continue
            print(f"| {name} | {c['retrieval_ms']} | {c['llm_calls']} | {c['prompt_tokens']} "
                  f"| {c['completion_tokens']} | {c['llm_ms']} |")


if __name__ == "__main__":
    main()
