import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make "import app..." work when running "python eval/no_context.py" from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402
from app.config import MODEL  # noqa: E402

# No-context test: ask every question to the LLM WITHOUT retrieval.
#
#   python eval/no_context.py kestrel      # reads eval/questions_kestrel.jsonl
#   python eval/no_context.py crime        # reads eval/questions_crime.jsonl
#
# If the model answers a question correctly with no excerpt at all, that question tests the model's
# training, not our RAG ("a RAG that answers questions the model already knew proves nothing").
#
# The script only COLLECTS the answers. Deciding which ones are correct is done by hand:
# open eval/no_context_<corpus>.json and set "known_without_rag" to true or false for each question.
# That is why an existing results file is never overwritten without --force (it holds your marks).

EVAL_DIR = Path(__file__).resolve().parent
LLM_DELAY_SECONDS = 4.5    # stay under the free-tier rate limit (~15 requests/minute)
PROVIDER_RETRIES = 3       # 429 / 503 happen on the free tier: wait and retry, then give up
RETRY_WAIT_SECONDS = 15

# The model gets the TOPIC, not the documents. Without it, "What is the full name of the main
# character?" is unanswerable for anyone, and we would underestimate what the model already knows.
TOPICS = {
    "crime": "the novel Crime and Punishment by Fyodor Dostoevsky",
    "kestrel": "Kestrel Analytics, a B2B dashboard and business intelligence software company",
}


# Same JSON contract as the RAG prompt (LLMAnswer), so "found": false is an explicit "I don't know".
def build_prompt(topic, question):
    return f"""Answer the question below from your own knowledge. It is about {topic}.

Rules:
- Answer in one or two sentences.
- If you do not know the answer, set "found" to false and say so in "answer". Do not guess.

Reply with a JSON object only, with exactly these keys:
{{"answer": string, "found": boolean, "citations": []}}

QUESTION:
{question}
"""


def ask(prompt):
    for attempt in range(1, PROVIDER_RETRIES + 1):
        try:
            return llm.ask_llm_for_answer(prompt), None
        except llm.ProviderError as e:
            if attempt == PROVIDER_RETRIES:
                return None, str(e)
            print(f"    {e} -> retry in {RETRY_WAIT_SECONDS}s")
            time.sleep(RETRY_WAIT_SECONDS)
        except llm.InvalidOutputError:
            return None, "invalid_output"


def main():
    parser = argparse.ArgumentParser(description="No-context test: the questions without retrieval")
    parser.add_argument("corpus", choices=sorted(TOPICS))
    parser.add_argument("--force", action="store_true", help="overwrite the results file (loses your marks)")
    args = parser.parse_args()

    questions_file = EVAL_DIR / f"questions_{args.corpus}.jsonl"
    out = EVAL_DIR / f"no_context_{args.corpus}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.name} already exists (it may hold your marks). Use --force to overwrite it.")
    if llm.client is None:
        raise SystemExit("This test needs GOOGLE_API_KEY: create .env from .env.example.")

    questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    topic = TOPICS[args.corpus]

    rows = []
    for q in questions:
        answer, error = ask(build_prompt(topic, q["question"]))
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "answerable": q["answerable"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "model_answer": answer.answer if answer else None,
            # The model's own "I don't know" (found = false). None when the call failed.
            "model_says_known": answer.found if answer else None,
            "error": error,
            # To fill BY HAND: true if the model's answer is correct without any retrieval.
            "known_without_rag": None,
        })
        print(f"{q['id']:4} {'ERROR ' + error if error else ('found   ' if answer.found else 'unknown ')} "
              f"{(answer.answer if answer else '')[:100]!r}")
        time.sleep(LLM_DELAY_SECONDS)

    out.write_text(json.dumps({
        "corpus": args.corpus,
        "model": MODEL,
        "date": f"{datetime.now():%Y-%m-%d %H:%M}",
        "topic_given": topic,
        "questions": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}"
          f" -> now set \"known_without_rag\" by hand for each question.")


if __name__ == "__main__":
    main()
