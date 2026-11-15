# Ask the documents — a RAG web app

Ask a question about a set of documents and get an answer **with its sources**: every answer cites the
passages it comes from, and you can open each one. Course project (EPITA, AI Systems Integration),
built in four versions: V1 naive RAG, V2 two corpora and evaluation, V3 hybrid search (embeddings + keywords +
concept graph), V4 concept map explorer and a cleaner UI.

Two document sets come with the app:
- **Kestrel Analytics**: the help centre of a fictional software company (product guide, billing policies,
  error codes, service catalog). Try *"What does error KST-2044 mean and how do I fix it?"*
- **Crime and Punishment**: the novel, with one invented character hidden in it (Denis Amselemovitch).

## Quick start

You need **Python 3.10 or newer** and **git**. Run every command from the project folder.

1. Get the code:

   ```bash
   git clone https://github.com/minkyu-shim/norag-rag.git
   cd norag-rag
   ```

2. Create a virtual environment and activate it (on Windows: `.venv\Scripts\activate`):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies. This takes a few minutes: it installs PyTorch for the local embedding model.

   ```bash
   pip install -r requirements.txt
   ```

4. Add your Google Gemini API key. Create a free key at https://aistudio.google.com/app/apikey, then:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and replace `your_google_api_key_here` with your key. Without a key the app still runs
   (documents, search, concept map), but it cannot write answers.

5. Start the app:

   ```bash
   uvicorn app.main:app
   ```

   Wait for `Application startup complete` in the terminal, then open http://127.0.0.1:8000.

**The first start takes about a minute:** the app copies the documents into `data/`, downloads the
embedding model (~90 MB, once) and indexes everything. Later starts take a few seconds.

## Using the app

Pick the document set at the top of the page, then:

- **Ask**: type a question. The answer cites its passages (`[C1]`, `[C2]`…): click one to read it.
  "Not in your documents" means the documents do not contain the answer; the app says so instead of guessing.
- **Concept map** (Kestrel): the teams, services, error codes, people and channels found in the documents,
  and how they are linked. Click a concept to see its links and the passage each one was read from, then
  **Ask about this** to ask a question about it.
- **Documents**: add your own `.md` or `.txt` files, or delete one. A new file is searchable right away.
- **Words**, **Evaluation**, **Settings**: tools for the course (word analytics, the evaluation of the
  retrieval, the chunk settings). See *How it works* below.

## Troubleshooting

- **`uvicorn: command not found`**: the virtual environment is not active. Run `source .venv/bin/activate`.
- **Port 8000 already in use**: start on another port, `uvicorn app.main:app --port 8001`, and open that port.
- **"The AI service did not answer"** (status `provider_error`): no key in `.env`, or the free tier of
  Gemini is busy (HTTP 429/503). Wait a minute and ask again.
- **The concept map says there is no map for Crime and Punishment**: only Kestrel has one. Building one costs
  one LLM call per chunk: `python scripts/build_graph.py crime` (about 1 600 calls).
- **Start from scratch**: stop the app and delete the `data/` folder. Everything is re-created at the next
  start (documents, indexes, concept map).

## How it works (for developers)

- Layout: `app/` (FastAPI backend), `web/` (plain HTML/JS, no build step), `eval/` (question sets and
  evaluation), `scripts/` (concept graph build), `docs/` (the seed documents and the seed concept graph),
  `data/` (everything created at runtime, not in git), `REPORT.md` (team, decision log, known failures).
- Seed files: on first start, `docs/CrimeAndPunishment.txt` and `docs/kestrel/*.md` are copied to
  `data/corpora/<corpus>/`, and `docs/graphs/kestrel.json` to `data/graph/kestrel.json`. After that, the
  Documents tab manages `data/corpora/`.
- Settings: chunk size, overlap and top-k (Settings tab, saved in `data/settings.json`). A chunk size or
  overlap change needs a **Re-index**; top-k does not.
- Without `GOOGLE_API_KEY`, `/api/ask` returns a clear `provider_error` and the retrieved passages.

### 1) Retrievers: dense, BM25, concept graph (V3)

Three retrievers, fused with Reciprocal Rank Fusion (RRF) over the ones that are switched on:

| Retriever | What it matches | Index |
|---|---|---|
| `dense` | meaning (embeddings, `all-MiniLM-L6-v2`) | ChromaDB, `data/chroma/` |
| `bm25` | words, identifiers such as `KST-2044` | SQLite FTS5, `data/bm25.sqlite` |
| `graph` | entities and the relations between them (Kestrel only) | `data/graph/kestrel.json` |

- **In the UI:** the *Retrievers* checkboxes of the Ask tab (and of the Evaluation tab). The default is dense + BM25. The *concept graph* box is greyed out until the graph is built.
- **In the evaluation:** `--retrievers`, a comma-separated list (see section 3).
- When the graph is on, the answer shows the **graph path**: the entities found in the question, the edges followed, and the chunks they come from.

### 2) Build the concept graph (V3, Kestrel)

The graph is built from the terminal only: one LLM call per chunk (about 40 for Kestrel, a few minutes on the free tier).

```bash
python scripts/build_graph.py kestrel
```

- The graph goes to `data/graph/kestrel.json`, and the build report (cost, dropped triples, merges) with a copy of the graph to `eval/results/graph_kestrel_c800_o10_<date>.json`.
- Extractions are cached per chunk in `data/graph/cache/`: a build stopped by a 429/503 error resumes where it stopped, and a second build costs no LLM call.
- Restart the app (or just reload the page) after the build: the *concept graph* checkbox becomes available.
- To ship a rebuilt graph with the repo, copy `data/graph/kestrel.json` to `docs/graphs/kestrel.json` (the seed used by a fresh clone).
- Extraction quality: `python eval/sample_triples.py kestrel` picks 10 random triples (fixed seed) and writes `eval/graph_triples_graded_kestrel.json`, where each one is graded by hand (`correct`, `wrong`, `unsupported`).

### 3) Evaluation

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

### 4) Words tab (V2)

Open the app, then the **Words** tab. Words are counted as the BM25 index sees them: lowercase, identifiers kept whole (`kst-2044`) plus their parts, stopwords left out, no stemming, in the chunks of the current chunk setting.

- **Corpus mode:** *Show vocabulary* lists every word with its number of occurrences and of chunks (most frequent, rarest, or alphabetical; with a filter). Click a word, or type one, to see the chunks that contain it, with the word highlighted.
- **Question mode:** type a question or pick one from the question set. For each of its words: how many chunks contain it, and, for a question of the set, whether it appears in the chunk that holds the answer. Words absent from the corpus and stopwords are listed apart.

## Course scripts (V0): NoRAG vs RAG

The two scripts the project started from, kept for comparison. They are independent from the web app.

### Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-2.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Default retrieval uses `n_results=5` for the question: `What does this say about Denis?`

### Project structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `docs/CrimeAndPunishment-shorter.txt`: shortened version of the main source text for quicker testing and rapid experimentation in Gemini online at https://gemini.google.com/app (See note below)
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `requirements.txt`: Python dependencies.

### Prerequisites

- Python 3.10+ (recommended)
- API key:
   - `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)


### Setup

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

### Run each script

With your virtual environment activated:

#### NoRAG + Google

```bash
python ask_norag_google.py
```

#### RAG + Google

```bash
python ask_rag_google.py
```

### Notes

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

### Free books from Project Gutenberg
- https://www.gutenberg.org/
- Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
- Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
