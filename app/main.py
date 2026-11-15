from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal
from app import documents, store, rag, graph, graph_retrieval, words, evaluation, llm, tracing
from app.config import WEB_DIR, CORPORA
from app.schemas import (Settings, SettingsResponse, AskRequest, AskResponse, DocumentInfo, CorpusName,
                         EvalRequest, WordsQuestionRequest)
from app.settings import load_settings, save_settings

# The web API. Both phases of RAG are behind it:
#   - indexing:  POST /api/documents (upload), POST /api/index (re-index everything)
#   - answering: POST /api/ask
# The UI (web/) is plain HTML/JS served by the same server, at http://127.0.0.1:8000/
#
# V2: every route works on ONE corpus, given explicitly (?corpus=kestrel, or "corpus" in the body).
# There is no default corpus on purpose: forgetting it is a 422, not a silent answer from the wrong corpus.
# The settings (chunk size, overlap, top-k) are shared: the same pipeline runs on both corpora.
#
# Run from the repo root:  uvicorn app.main:app --reload
#
# Note: the routes are plain "def" (not "async def"): FastAPI runs them in a thread pool,
# so a slow embedding or LLM call does not block the whole server.


@asynccontextmanager
async def lifespan(app):
    # First start: copy the seed files of every corpus into data/corpora/ and index what is
    # not indexed yet, so the app is usable right away (takes ~20s for Crime and Punishment).
    tracing.purge_old_traces()      # V4: retention (see TRACE_RETENTION_DAYS in app/config.py)
    documents.seed_all()
    for corpus in CORPORA:
        documents.index_missing(corpus, load_settings())
    yield


app = FastAPI(title="Hybrid RAG — V3", lifespan=lifespan)


def settings_response(corpus, settings):
    # The index is "ready" when every document of the corpus has chunks in the collection of these settings.
    ready = all(documents.is_indexed(corpus, p.name, settings) for p in documents.list_document_paths(corpus))
    return SettingsResponse(
        **settings.model_dump(),
        corpus=corpus,
        index_ready=ready,
        collection=store.collection_name(corpus, settings),
    )


# --- Corpora -----------------------------------------------------------------
@app.get("/api/corpora", response_model=list[str])
def list_corpora():
    return list(CORPORA)


# --- Settings ----------------------------------------------------------------
@app.get("/api/settings", response_model=SettingsResponse)
def get_settings(corpus: CorpusName):
    return settings_response(corpus, load_settings())


@app.put("/api/settings", response_model=SettingsResponse)
def put_settings(settings: Settings, corpus: CorpusName):
    # Pydantic already checked the ranges (see Settings): a bad value is a 422, not a crash.
    # Saving does NOT re-index: if the chunking changed, index_ready will be False.
    # The settings are saved for all corpora; the response tells the index state of `corpus`.
    save_settings(settings)
    return settings_response(corpus, settings)


@app.post("/api/index")
def reindex(corpus: CorpusName):
    return {"corpus": corpus, "indexed": documents.index_all(corpus, load_settings())}


# --- Concept graph (V3) -----------------------------------------------------------
@app.get("/api/graph")
def graph_info(corpus: CorpusName):
    # Built from the terminal only (python scripts/build_graph.py <corpus>): it takes minutes and LLM calls.
    if not graph_retrieval.is_available(corpus):
        return {"corpus": corpus, "available": False}
    data = graph.load(corpus)
    return {"corpus": corpus, "available": True, "built": data["built"], "settings": data["settings"],
            "nodes": len(data["nodes"]), "edges": len(data["edges"])}


# --- Documents ---------------------------------------------------------------
@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents(corpus: CorpusName):
    settings = load_settings()
    collection = store.get_collection(corpus, settings)
    return [
        DocumentInfo(name=p.name, size_bytes=p.stat().st_size, chunks=store.count_chunks(collection, p.name))
        for p in documents.list_document_paths(corpus)
    ]


@app.post("/api/documents")
def upload_document(file: UploadFile, corpus: CorpusName):
    try:
        documents.save_document(corpus, file.filename, file.file.read())
    except documents.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    chunks = documents.index_document(corpus, file.filename, load_settings())
    return {"corpus": corpus, "name": file.filename, "chunks": chunks}


@app.delete("/api/documents/{name}")
def delete_document(name: str, corpus: CorpusName):
    try:
        documents.delete_document(corpus, name)
    except documents.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"corpus": corpus, "deleted": name}


# --- Ask ---------------------------------------------------------------------
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        response, http_status = rag.answer(
            request.corpus, request.question, request.sources, load_settings(), request.retrievers)
    except rag.NotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Even on an error status (429, 502...) the body is a full AskResponse,
    # so the UI can show the message AND the chunks that were retrieved.
    return JSONResponse(status_code=http_status, content=response.model_dump())


# --- Traces (V4, Part A) -----------------------------------------------------------
# One trace per question / indexing run, one span per step. The answer carries its trace_id:
# the "Details" button of the UI opens GET /api/traces/{trace_id}. Stored in data/traces.sqlite, on this machine.
@app.get("/api/traces")
def list_traces(limit: int = 50):
    return tracing.list_traces(min(max(limit, 1), 500))


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str):
    trace = tracing.get_trace(trace_id)
    if trace is None:
        # Traces are deleted after TRACE_RETENTION_DAYS days, or the data/ folder was wiped.
        raise HTTPException(status_code=404, detail="This trace does not exist (any more).")
    return trace


# --- Evaluation (V2: also startable from the UI) ------------------------------------
# Same code as the terminal (python eval/run_eval.py), same results folder. Uses the CURRENT settings.
# Synchronous on purpose (simple): retrieval only takes seconds; with the LLM the request lasts ~2 min.
@app.post("/api/eval")
def run_evaluation(request: EvalRequest):
    if request.with_llm and llm.client is None:
        raise HTTPException(status_code=400, detail="Generating answers needs GOOGLE_API_KEY in .env.")
    if "graph" in request.retrievers and not graph_retrieval.is_available(request.corpus):
        raise HTTPException(status_code=400, detail=f"No concept graph for {request.corpus!r}: "
                                                    f"run  python scripts/build_graph.py {request.corpus}")
    result = evaluation.evaluate(request.corpus, load_settings(), retrievers=request.retrievers,
                                 with_llm=request.with_llm, log=lambda line: None)
    path = evaluation.save(result)
    # The chunk texts stay in the file: the UI only needs the numbers and one line per question.
    questions = [{"id": r["id"], "type": r["type"], "answerable": r["answerable"], "question": r["question"],
                  "rank": r.get("rank"), "llm_status": r["llm"]["status"] if "llm" in r else None}
                 for r in result["questions"]]
    return {"file": str(path.relative_to(evaluation.ROOT_DIR)), "settings": result["settings"],
            "summary": result["summary"], "questions": questions}


# --- Words analytics (V2) ---------------------------------------------------------
# Corpus mode: vocabulary, and word -> chunks. Question mode: question -> words. See app/words.py.
def words_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except words.WordsNotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/words/vocabulary")
def words_vocabulary(corpus: CorpusName, contains: str = "", order: Literal["frequent", "rare", "alpha"] = "frequent",
                     limit: int = 200):
    return words_call(words.vocabulary, corpus, load_settings(), contains, order, min(max(limit, 1), 2000))


@app.get("/api/words/word")
def words_word(corpus: CorpusName, word: str):
    return words_call(words.word_chunks, corpus, load_settings(), word)


@app.get("/api/words/questions")
def words_questions(corpus: CorpusName):
    return words.question_set(corpus)


@app.post("/api/words/question")
def words_question(request: WordsQuestionRequest):
    if not request.question.strip() and not request.question_id:
        raise HTTPException(status_code=422, detail="Type a question or pick one from the question set.")
    return words_call(words.question_words, request.corpus, load_settings(), request.question, request.question_id)


# The UI. Mounted last, so that it does not hide the /api routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
