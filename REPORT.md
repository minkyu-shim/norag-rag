# REPORT — V1 (Naive RAG app)

> TODO (team): the decision log must be **written by you**. The entries below hold the facts
> (options, choice, numbers measured on Oct. 28th). Rewrite every "Why" in your own words,
> and delete this note.

## 1. Team details

| Member | Worked on |
|---|---|
| TODO | TODO |
| TODO | TODO |

## 2. Decision log

### Corpus — *Crime and Punishment* + the Denis needle
- **Options:** Crime and Punishment (already in the repo); several Gutenberg books; our own documents.
- **Picked:** Crime and Punishment (1.18 MB, ~1600 chunks at 800 chars).
- **Why:** TODO. Facts: it is large enough that retrieval matters. The LLM already knows the book, so most questions *could* be answered without retrieval. The injected Denis Amselemovitch passage (line ~14613) is the only content the LLM cannot know.

### Stack — FastAPI + plain HTML/JS
- **Options:** FastAPI + vanilla JS; FastAPI + React (Vite); Streamlit.
- **Picked:** FastAPI (backend, Pydantic built in) + one static page, no build step.
- **Why:** TODO. Facts: the spec asks for the pipeline "behind a web API". Streamlit has no separate API layer.

### Embedding model — `all-MiniLM-L6-v2` (local)
- **Options:** MiniLM (local, 384 dims); a bigger local model; an API embedding model (Gemini).
- **Picked:** MiniLM, same as the reference script.
- **Why:** TODO. Facts: no API cost or quota, and indexing the whole book takes ~20 s on a laptop CPU. The input limit is 256 word-pieces (~1000 chars), so long chunks get truncated. Changing the model means deleting `data/chroma/`.

### Vector DB — ChromaDB, one collection per (chunk size, overlap)
- **Options:** one collection, wiped on every settings change; one collection per setting.
- **Picked:** one per setting (`chunks_800_10`, `chunks_500_20`, ...).
- **Why:** TODO. Facts: chunks from different settings are never mixed, switching back to an already indexed setting is free, and top-k needs no re-index. Cost: disk space, and a delete has to clean every collection.

### Chunking — fixed 800-char windows, 10 % overlap
- **Options:** fixed windows (lab); paragraphs (reference script); sentence-aware.
- **Picked:** fixed windows, the lab setting. **Naive on purpose:** V1 is the baseline.
- **Why:** TODO. Every chunk stores `start`/`end` offsets and line numbers, so a reference can be checked in the file.

### LLM output — JSON mode + Pydantic `LLMAnswer {answer, found, citations}`
- **Options:** free text; JSON mode + validation; native structured output.
- **Picked:** `response_format=json_object`, then `LLMAnswer.model_validate_json`. One retry, then status `invalid_output`.
- **Why:** TODO. `found` makes "I don't know" a field we can test rather than a sentence to parse.

### Citations — short labels `C1..Ck`, checked on the server
- Excerpts are labelled `C1..Ck` in the prompt. A citation that is not one of the retrieved labels is dropped with a warning, and if nothing valid remains the status is `unverified`.
- **Why:** TODO (answers "how do you know a cited reference was actually retrieved?").

### Controlled failures — one explicit status per outcome
`answered | not_found | unverified | invalid_output | provider_error`, plus HTTP 400 for "unknown document / not indexed" and 422 for invalid input. The retrieved chunks are always returned. No retries on 429 (`max_retries=0`): the user is told right away.

### Evaluation — evidence quotes, not chunk ids
- 25 questions in `eval/questions.json`: 10 easy, 8 hard (paraphrase, no character name, vocabulary mismatch), 2 needle, 5 unanswerable.
- The ground truth is a quote, located in the file as a character span. A hit means a retrieved chunk covers ≥ 50 % of the span, so the measure works for any chunk size.
- **Why / how many questions is enough:** TODO.

## 3. Baseline numbers (retrieval only, `python eval/run_eval.py`)

| Setting | n | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|---|
| chunk 800, overlap 10 % | 20 | 0.15 | 0.25 | 0.25 | 0.20 |
| chunk 500, overlap 20 % | 20 | 0.10 | 0.20 | 0.25 | 0.16 |

Mean top-1 cosine distance (800/10): answerable **0.450**, unanswerable **0.481**. These are almost the same, so a distance threshold alone would not separate "nothing relevant" from a real hit.

## 4. Known failures

| # | Question | What we observed | Likely cause |
|---|---|---|---|
| 1 | *What is the full name of the main character?* (e01) | The evidence chunk is not even in the top 200 | The question shares no rare word with the passage; MiniLM matches on topic, not on "name" |
| 2 | *Which character might be a time traveller who used AI to get into the story?* (n02) | The needle chunk ranks 8th (outside the top 5), while *Who is Denis Amselemovitch?* ranks 1st | Paraphrase/vocabulary mismatch: the query says "time traveller", the text says "artifact from the future" |
| 3 | *What document did the drunk official's daughter have to carry…?* (h08) | The evidence ("yellow ticket") ranks 151st | The hard questions avoid character names, and without names MiniLM retrieval falls apart (hard hit@5 = 0.12) |
| 4 | *What exactly does the hero say when he gives himself up?* (h05) | The confession is not in the top 200 | A single query, and many chunks talk about confession and the murder; the key sentence is diluted in an 800-char window |
| 5 | *What is the capital of Australia?* (u02) | 5 chunks are still retrieved and sent to the LLM | Top-k is always returned, with no relevance threshold (weak spot #4) |
| 6 | Needle passage | Split across two chunks (`::1024` = lines 14600–14614, `::1025` = lines 14613–14631) | Fixed windows ignore paragraph boundaries |
| 7 | TODO: LLM end-to-end (`run_eval.py --llm`) | TODO: not run yet (no API key set when the baseline was measured) | |
