# norag-rag project

This workspace contains 2 Python applications to compare **NoRAG** (full-context prompting) versus **RAG** (retrieval-augmented generation) using **Google Gemini** via the **OpenAI SDK**.

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-2.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Default retrieval uses `n_results=5` for the question: `What does this say about Denis?`

## Project Structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `docs/CrimeAndPunishment-shorter.txt`: shortened version of the main source text for quicker testing and rapid experimentation in Gemini online at https://gemini.google.com/app (See note below)
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+ (recommended)
- API key:
   - `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)


## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set your real keys.

## Run Each Application

With your virtual environment activated:

### 1) NoRAG + Google

```bash
python ask_norag_google.py
```

### 2) RAG + Google

```bash
python ask_rag_google.py
```

### 3) The RAG web app (V1 → V3)

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/

- The app has two corpora, picked in the header: `crime` (our own) and `kestrel` (the course's knowledge base). On first start, their seed files (`docs/CrimeAndPunishment.txt` and `docs/kestrel/*.md`) are copied into `data/corpora/<corpus>/` and indexed (~20 s for the book; the embedding model downloads the very first time).
- **Ask** tab: choose the documents, switch the retrievers on or off, ask a question, get the answer with its references. Click a reference to see the chunk it points to.
- **Documents** tab: upload (`.md` / `.txt`) or delete documents.
- **Words** tab (V2): explore the words of the corpus, as the BM25 index sees them (see below).
- **Evaluation** tab (V2): run the evaluation of the current corpus from the browser (see below).
- **Settings** tab: chunk size, overlap and top-k. A chunk size or overlap change needs a **Re-index** (button in the UI); top-k does not.
- Without `GOOGLE_API_KEY`, everything works except the LLM answer: `/api/ask` returns a clear `provider_error`.
- Layout: `app/` (FastAPI backend), `web/` (plain HTML/JS UI), `eval/` (question sets + evaluation), `scripts/` (graph build), `data/` (runtime data, git-ignored), `REPORT.md` (team, decision log, known failures).

### 4) Retrievers: dense, BM25, concept graph (V3)

Three retrievers, fused with Reciprocal Rank Fusion (RRF) over the ones that are switched on:

| Retriever | What it matches | Index |
|---|---|---|
| `dense` | meaning (embeddings, `all-MiniLM-L6-v2`) | ChromaDB, `data/chroma/` |
| `bm25` | words, identifiers such as `KST-2044` | SQLite FTS5, `data/bm25.sqlite` |
| `graph` | entities and the relations between them (Kestrel only) | `data/graph/kestrel.json` |

- **In the UI:** the *Retrievers* checkboxes of the Ask tab (and of the Evaluation tab). The default is dense + BM25. The *concept graph* box is greyed out until the graph is built.
- **In the evaluation:** `--retrievers`, a comma-separated list (see section 6).
- When the graph is on, the answer shows the **graph path**: the entities found in the question, the edges followed, and the chunks they come from.

### 5) Build the concept graph (V3, Kestrel)

The graph is built from the terminal only: one LLM call per chunk (about 40 for Kestrel, a few minutes on the free tier).

```bash
python scripts/build_graph.py kestrel
```

- The graph goes to `data/graph/kestrel.json`, and the build report (cost, dropped triples, merges) with a copy of the graph to `eval/results/graph_kestrel_c800_o10_<date>.json`.
- Extractions are cached per chunk in `data/graph/cache/`: a build stopped by a 429/503 error resumes where it stopped, and a second build costs no LLM call.
- Restart the app (or just reload the page) after the build: the *concept graph* checkbox becomes available.
- Extraction quality: `python eval/sample_triples.py kestrel` picks 10 random triples (fixed seed) and writes `eval/graph_triples_graded_kestrel.json`, where each one is graded by hand (`correct`, `wrong`, `unsupported`).

### 6) Evaluation

The question sets are `eval/questions_kestrel.jsonl` and `eval/questions_crime.jsonl` (Kestrel format: each answerable question gives its evidence as a source file and a quote). A retrieved chunk is relevant when it covers at least half of an evidence quote; unanswerable questions are left out of hit@k and recall@k.

**From the terminal** (one command per configuration):

```bash
python eval/run_eval.py kestrel                                   # dense + BM25, chunk 800, overlap 10%, k = 1,3,5,10
python eval/run_eval.py kestrel --retrievers dense                # one retriever only
python eval/run_eval.py kestrel --retrievers dense,bm25,graph     # with the concept graph (build it first)
python eval/run_eval.py crime --chunk-size 500 --overlap 20       # another chunk setting (indexed if needed)
python eval/run_eval.py kestrel --llm                             # also generate the answers (1 API call per question)
```

**From the browser:** the **Evaluation** tab. Pick the retrievers, tick *Also generate the answers with the LLM* if you want them, and click *Run the evaluation*. It uses the corpus of the header and the current chunk settings.

Both write the same file: `eval/results/<corpus>_<retrievers>_c<size>_o<overlap>[_llm]_<date>.json`. With the LLM, every answer is saved next to the chunks it was given, with `"grade": null` to fill by hand (`correct`, `partial`, `wrong`, `unsupported`).

Other evaluation scripts:

```bash
python eval/compare.py eval/results/kestrel_*_c800_o10_*.json --metrics hit@5,recall@5   # markdown tables, configurations × question types
python eval/no_context.py kestrel                                                        # the no-context test: questions asked WITHOUT retrieval
```

### 7) Words tab (V2)

Open the app, then the **Words** tab. Words are counted as the BM25 index sees them: lowercase, identifiers kept whole (`kst-2044`) plus their parts, stopwords left out, no stemming, in the chunks of the current chunk setting.

- **Corpus mode:** *Show vocabulary* lists every word with its number of occurrences and of chunks (most frequent, rarest, or alphabetical; with a filter). Click a word, or type one, to see the chunks that contain it, with the word highlighted.
- **Question mode:** type a question or pick one from the question set. For each of its words: how many chunks contain it, and, for a question of the set, whether it appears in the chunk that holds the answer. Words absent from the corpus and stopwords are listed apart.

### 8) Traces and personal data (V4)

**Traces.** Every question and every indexing run (upload, re-index, graph build) is recorded as one *trace*, with one *span* per step (each retriever, the RRF fusion, the prompt, each LLM call with its tokens and time, the output check, the citation check).

- **In the UI:** under every answer, click **Details**. A side panel shows the outcome, total time, LLM calls, tokens, cost, how many personal items were hidden, and the steps as a timeline (click a step for its details).
- **Through the API:** `GET /api/traces/{trace_id}` (the `trace_id` is in the answer) and `GET /api/traces?limit=50` (latest first). The graph build prints its trace id at the end.
- **Where:** `data/traces.sqlite`, on your machine only, deleted after 30 days (`TRACE_RETENTION_DAYS` in `app/config.py`). Nothing is sent to a tracing service.
- **What is never stored:** API keys, whole documents (a trace holds chunk ids; every text is cut at 600 characters), and personal data (see below).
- **Cost in dollars:** set `LLM_PRICE_INPUT_PER_MTOK` and `LLM_PRICE_OUTPUT_PER_MTOK` (USD per million tokens, from your provider's price page) in `.env`. Until then, the panel shows tokens and "n/a".

**Personal data redaction.** E-mail addresses of people, phone numbers and names (when a cue says it is a person: "my name is …", "Dr …", "Dear …", or listed in `data/redaction_names.txt`, one per line) are replaced by `[EMAIL_1]`, `[PHONE_1]`, `[PERSON_1]` before a prompt is sent to the LLM provider and before anything is written to a trace. The answer comes back with your own data put back, so you read the original on your machine. Role addresses (`privacy@…`), IP addresses, dates and error codes are left alone. Switch it off for a comparison with `REDACT_PERSONAL_DATA=0` in `.env`.

```bash
python -m pytest tests -q        # tracing, redaction and an end-to-end ask with a fake LLM (no API key, no network)
```

## Notes

- Both applications use the OpenAI SDK to interact with Google Gemini via Google's OpenAI-compatible endpoint.
- The RAG app indexes text chunks into `chroma_db/` and reuses them across runs.
- First RAG run may take longer due to indexing.
- Debug prompt printing can be enabled by passing `debug=True` in the NoRAG app.
- RAG app prints retrieved context for verification.
- Streaming output is enabled by default in both applications (controlled by `STREAM_ON` variable).

- Using the shortened version of the text (`docs/CrimeAndPunishment-shorter.txt`)
	- Go online at https://gemini.google.com/app
	- At the prompt, start writing "Here is a story: ",
	- Then paste the content of docs/CrimeAndPunishment-shorter.txt into the Gemini chat app and then, at the end of the prompt, ask "What can you tell me about Denis Amselemovitch?"
	- Observe the response from Gemini.

# Free books from Project Gutenberg
- https://www.gutenberg.org/
# Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
# Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
