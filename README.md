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

### 3) V1 — Naive RAG web app

```bash
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/

- On first start, `docs/CrimeAndPunishment.txt` is copied into `data/docs/` and indexed (~20 s; the embedding model downloads the very first time).
- **Ask** tab: choose the documents, ask a question, get the answer with its references. Click a reference to see the chunk it points to.
- **Documents** tab: upload (`.md` / `.txt`) or delete documents.
- **Settings** tab: chunk size, overlap and top-k. A chunk size or overlap change needs a **Re-index** (button in the UI); top-k does not.
- Without `GOOGLE_API_KEY`, everything works except the LLM answer: `/api/ask` returns a clear `provider_error`.
- Layout: `app/` (FastAPI backend), `web/` (plain HTML/JS UI), `eval/` (question set + retrieval eval), `data/` (runtime data, git-ignored), `REPORT.md` (team, decision log, known failures).

### 4) Retrieval evaluation

```bash
python eval/run_eval.py                                # chunk 800, overlap 10%, hit@1/3/5 + MRR
python eval/run_eval.py --chunk-size 500 --overlap 20  # compare another setting
python eval/run_eval.py --llm                          # also run the LLM (about 25 API calls)
```

Results are printed and saved in `eval/results/`. Ground truth is a quote of the evidence (character span in the file), so numbers stay comparable when the chunk size changes.

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
