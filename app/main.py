from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app import documents, store, rag
from app.config import WEB_DIR
from app.schemas import Settings, SettingsResponse, AskRequest, AskResponse, DocumentInfo
from app.settings import load_settings, save_settings

# The web API. Both phases of RAG are behind it:
#   - indexing:  POST /api/documents (upload), POST /api/index (re-index everything)
#   - answering: POST /api/ask
# The UI (web/) is plain HTML/JS served by the same server, at http://127.0.0.1:8000/
#
# Run from the repo root:  uvicorn app.main:app --reload
#
# Note: the routes are plain "def" (not "async def"): FastAPI runs them in a thread pool,
# so a slow embedding or LLM call does not block the whole server.


@asynccontextmanager
async def lifespan(app):
    # First start: copy the book into data/docs/ and index what is not indexed yet,
    # so the app is usable right away (takes ~20s for Crime and Punishment).
    documents.seed_corpus()
    settings = load_settings()
    for path in documents.list_document_paths():
        if not documents.is_indexed(path.name, settings):
            documents.index_document(path.name, settings)
    yield


app = FastAPI(title="Naive RAG — V1", lifespan=lifespan)


def settings_response(settings):
    # The index is "ready" when every document has chunks in the collection of these settings.
    ready = all(documents.is_indexed(p.name, settings) for p in documents.list_document_paths())
    return SettingsResponse(
        **settings.model_dump(),
        index_ready=ready,
        collection=store.collection_name(settings.chunk_size, settings.overlap_percent),
    )


# --- Settings ----------------------------------------------------------------
@app.get("/api/settings", response_model=SettingsResponse)
def get_settings():
    return settings_response(load_settings())


@app.put("/api/settings", response_model=SettingsResponse)
def put_settings(settings: Settings):
    # Pydantic already checked the ranges (see Settings): a bad value is a 422, not a crash.
    # Saving does NOT re-index: if the chunking changed, index_ready will be False.
    save_settings(settings)
    return settings_response(settings)


@app.post("/api/index")
def reindex():
    return {"indexed": documents.index_all(load_settings())}


# --- Documents ---------------------------------------------------------------
@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents():
    settings = load_settings()
    collection = store.get_collection(settings.chunk_size, settings.overlap_percent)
    return [
        DocumentInfo(name=p.name, size_bytes=p.stat().st_size, chunks=store.count_chunks(collection, p.name))
        for p in documents.list_document_paths()
    ]


@app.post("/api/documents")
def upload_document(file: UploadFile):
    try:
        documents.save_document(file.filename, file.file.read())
    except documents.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    chunks = documents.index_document(file.filename, load_settings())
    return {"name": file.filename, "chunks": chunks}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    try:
        documents.delete_document(name)
    except documents.DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": name}


# --- Ask ---------------------------------------------------------------------
@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        response, http_status = rag.answer(request.question, request.sources, load_settings())
    except rag.NotReadyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Even on an error status (429, 502...) the body is a full AskResponse,
    # so the UI can show the message AND the chunks that were retrieved.
    return JSONResponse(status_code=http_status, content=response.model_dump())


# The UI. Mounted last, so that it does not hide the /api routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
