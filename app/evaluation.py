import json
import time
from datetime import datetime
from app import documents, rag
from app.schemas import Cost, DEFAULT_RETRIEVERS
from app.chunking import find_span, coverage
from app.config import ROOT_DIR, MODEL

# Retrieval evaluation: how often is the RIGHT PASSAGE in the top-k?
# Shared by the terminal (eval/run_eval.py) and the API (POST /api/eval), so both measure the same thing.
#
# The question sets are eval/questions_<corpus>.jsonl (Kestrel format). Each answerable question
# lists its evidence: a source file and a short QUOTE. We find the quote in the file to get its
# character span [start, end). A retrieved chunk is RELEVANT when it comes from that source and
# covers at least MIN_COVERAGE of that span. Chunk ids change with every setting, character offsets
# do not, so the numbers stay comparable across chunk sizes.
#
# Two measures, on the top k chunks, for every k in ks:
#   - hit@k:    the question has at least one relevant chunk in the top k (HitRate@k). The main one.
#   - recall@k: share of the question's evidence quotes covered in the top k. Only differs from
#               hit@k for questions with several quotes (split answers), where one hit is not enough.
# Unanswerable questions have no relevant chunk: they are left out of both measures.
#
# Retrieval only by default: free, fast and deterministic. with_llm=True also asks the LLM and saves
# each answer next to the chunks it was given, ready to be graded by hand ("grade": null).
#
# V3: the same questions are run with any set of retrievers (dense, bm25, graph), fused with RRF.
#   - hit@k / recall@k are measured on the FUSED list (what the LLM would get)
#   - per_retriever: hit@k of each retriever's OWN list, before fusion, measured in the same run
#   - lost@k: questions where a retriever had the evidence in ITS top k but the fused top k did not
#     (it was pushed out by the fusion: this answers "did the graph's chunks reach the top k?")
#   - cost per question (retrieval time; LLM calls, tokens, time with with_llm)
#   - unanswerable questions (with with_llm): correct abstention ("not_found") vs hallucination

EVAL_DIR = ROOT_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
MIN_COVERAGE = 0.5
LLM_DELAY_SECONDS = 4.5   # stay under the free-tier rate limit (~15 requests/minute)
DEFAULT_KS = [1, 3, 5, 10]


def questions_file(corpus):
    return EVAL_DIR / f"questions_{corpus}.jsonl"


def load_questions(corpus):
    lines = questions_file(corpus).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def covers(chunk, evidence):
    return chunk.source == evidence["source"] and coverage(chunk, evidence["span"]) >= MIN_COVERAGE


def first_hit_rank(retrieved, evidence_list):
    # Rank (1 = first) of the first chunk that covers ANY of the evidence quotes. None = no hit.
    for rank, chunk in enumerate(retrieved, start=1):
        if any(covers(chunk, ev) for ev in evidence_list):
            return rank
    return None


def evidence_ranks(retrieved, evidence_list):
    # For EACH evidence quote, the rank of the first chunk that covers it (None = not retrieved).
    return [first_hit_rank(retrieved, [ev]) for ev in evidence_list]


def evaluate(corpus, settings, ks=DEFAULT_KS, retrievers=DEFAULT_RETRIEVERS, with_llm=False, log=print):
    """Runs every question of the corpus' question set. Returns the result dict (see save())."""
    ks = sorted(ks)
    questions = load_questions(corpus)
    # Retrieve max(ks) chunks to measure every k at once. The LLM only gets the first top_k of them.
    retrieval_settings = settings.model_copy(update={"top_k": ks[-1]})

    # Make sure every document of the corpus is indexed with THIS setting (same code as the app).
    documents.seed_corpus(corpus)
    documents.index_missing(corpus, settings)
    sources = [p.name for p in documents.list_document_paths(corpus)]
    texts = {name: documents.read_document(corpus, name) for name in sources}

    rows = []
    for q in questions:
        # Every question searches the WHOLE corpus, like a real user: knowing the right file is part of the job.
        started = time.perf_counter()
        retrieved, rankings = rag.retrieve(corpus, q["question"], sources, retrieval_settings, retrievers)
        cost = Cost(retrieval_ms=round((time.perf_counter() - started) * 1000, 1))
        evidence = [{**ev, "span": find_span(texts[ev["source"]], ev["quote"])} for ev in q["evidence"]]
        row = {
            "id": q["id"],
            "type": q["type"],
            "answerable": q["answerable"],
            "question": q["question"],
            "top1_distance": round(retrieved[0].distance, 4) if retrieved and retrieved[0].distance is not None else None,
        }
        if q["answerable"]:
            row["rank"] = first_hit_rank(retrieved, evidence)
            row["evidence_ranks"] = evidence_ranks(retrieved, evidence)
            # The rank of the evidence in each retriever's own list (up to RRF_CANDIDATES), BEFORE fusion.
            row["retriever_ranks"] = {name: first_hit_rank(chunks, evidence) for name, chunks in rankings.items()}
        # The chunks themselves are saved: the grader checks answers against them, and the Words tab
        # diagnosis starts from them.
        row["retrieved"] = [
            {"label": c.label, "id": c.id, "source": c.source, "lines": f"{c.start_line}-{c.end_line}",
             "found_by": c.found_by, "rrf_score": c.rrf_score,
             "distance": round(c.distance, 4) if c.distance is not None else None, "bm25_score": c.bm25_score,
             "relevant": any(covers(c, ev) for ev in evidence), "text": c.text}
            for c in retrieved
        ]

        if with_llm:
            given = retrieved[:settings.top_k]
            response, _ = rag.answer_from_chunks(q["question"], given, retrievers, cost)
            row["llm"] = {
                "given_chunks": [c.label for c in given],
                "status": response.status,
                "answer": response.answer,
                "citations": [c.label for c in response.citations],
                # Did a cited chunk cover the evidence? (automatic hint, NOT the grade)
                "cited_evidence": first_hit_rank(response.citations, evidence) is not None if evidence else None,
                "expected_answer": q["expected_answer"],
                # To fill BY HAND: correct / partial / wrong / unsupported, and for a miss, its cause.
                "grade": None,
                "failure": None,          # "retrieval" or "generation"
                "grading_note": None,
            }
            time.sleep(LLM_DELAY_SECONDS)
        row["cost"] = cost.model_dump()

        log(f"{q['id']:4} {q['type']:17} rank={row.get('rank', '-')}"
            + (f"  llm={row['llm']['status']}" if with_llm else ""))
        rows.append(row)

    return {
        "corpus": corpus,
        "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        "settings": {**settings.model_dump(), "retrievers": retrievers, "rrf_k": rag.RRF_K,
                     "rrf_candidates": rag.RRF_CANDIDATES, "ks": ks, "min_coverage": MIN_COVERAGE,
                     "with_llm": with_llm, "model": MODEL if with_llm else None},
        "summary": summarize(rows, ks, retrievers, with_llm),
        "questions": rows,
    }


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows, ks, retrievers, with_llm):
    answerable = [r for r in rows if r["answerable"]]
    # Overall AND per type: an average hides the type that always fails.
    groups = {"ALL": answerable}
    for t in sorted({r["type"] for r in answerable}):
        groups[t] = [r for r in answerable if r["type"] == t]

    summary = {}
    for name, group in groups.items():
        line = {"n": len(group)}
        for k in ks:
            line[f"hit@{k}"] = round(sum(1 for r in group if r["rank"] and r["rank"] <= k) / len(group), 3)
        for k in ks:
            recalls = [sum(1 for rank in r["evidence_ranks"] if rank and rank <= k) / len(r["evidence_ranks"])
                       for r in group]
            line[f"recall@{k}"] = round(sum(recalls) / len(group), 3)
        summary[name] = line

    # V3: each retriever on its own (before fusion), and what the fusion lost, per question type.
    summary["per_retriever"] = {}
    for name in retrievers:
        per_type = {}
        for group_name, group in groups.items():
            line = {"n": len(group)}
            for k in ks:
                own = [r for r in group if r["retriever_ranks"][name] and r["retriever_ranks"][name] <= k]
                line[f"hit@{k}"] = round(len(own) / len(group), 3)
                # Found in the retriever's own top k, but not in the fused top k.
                line[f"lost@{k}"] = [r["id"] for r in own if not (r["rank"] and r["rank"] <= k)]
            per_type[group_name] = line
        summary["per_retriever"][name] = per_type

    # Is "nothing relevant" visible in the scores? (lower distance = closer)
    unanswerable = [r for r in rows if not r["answerable"]]
    summary["top1_distance"] = {
        "answerable": mean([r["top1_distance"] for r in answerable]),
        "unanswerable": mean([r["top1_distance"] for r in unanswerable]),
    }

    # V3: cost per question, averaged over all the questions of this configuration.
    summary["cost_per_question"] = {key: mean([r["cost"][key] for r in rows]) for key in Cost.model_fields}

    # V3: unanswerable questions, with the LLM: did it abstain or make something up?
    #   not_found            -> correct abstention
    #   answered/unverified  -> hallucination (an answer to a question the corpus cannot answer)
    #   invalid_output/provider_error -> no answer at all, counted apart (not the model's fault)
    if with_llm:
        statuses = [r["llm"]["status"] for r in unanswerable]
        summary["unanswerable"] = {
            "n": len(statuses),
            "abstained": statuses.count("not_found"),
            "hallucinated": statuses.count("answered") + statuses.count("unverified"),
            "errors": statuses.count("invalid_output") + statuses.count("provider_error"),
        }
    return summary


def config_label(retrievers):
    # "dense+bm25": used in the results file name and in eval/compare.py.
    return "+".join(retrievers)


def save(result):
    RESULTS_DIR.mkdir(exist_ok=True)
    s = result["settings"]
    name = (f"{result['corpus']}_{config_label(s['retrievers'])}_c{s['chunk_size']}_o{s['overlap_percent']}"
            f"{'_llm' if s['with_llm'] else ''}_{datetime.now():%Y%m%d-%H%M%S}.json")
    path = RESULTS_DIR / name
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
