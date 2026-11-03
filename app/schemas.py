from typing import Literal
from pydantic import BaseModel, Field
from app.config import CORPORA

# Every piece of data that crosses a boundary (UI <-> API, LLM -> API) has a Pydantic schema.
# FastAPI uses them to validate requests, and we use LLMAnswer to validate what the LLM returns
# BEFORE it reaches the UI: an LLM answer that does not match the schema is never shown as an answer.

# The corpus names, built from app/config.py so the list lives in one place.
# Literal["crime", "kestrel"]: an unknown corpus is a 422 from FastAPI, before our code runs.
CorpusName = Literal[tuple(CORPORA)]


# --- Settings the user can change from the UI ---------------------------------
class Settings(BaseModel):
    chunk_size: int = Field(ge=100, le=4000)
    overlap_percent: int = Field(ge=0, le=50)
    top_k: int = Field(ge=1, le=20)


class SettingsResponse(Settings):
    # The settings are shared by all corpora, but the index is not: these fields are for ONE corpus.
    corpus: str
    # False when no index exists yet for this (corpus, chunk_size, overlap): the user must re-index.
    index_ready: bool
    collection: str


# --- What the LLM must return ------------------------------------------------
class LLMAnswer(BaseModel):
    answer: str
    # False when the excerpts do not contain the answer ("I don't know").
    found: bool
    # Labels of the excerpts used, e.g. ["C1", "C3"]. Checked against what was retrieved.
    citations: list[str] = []


# --- Ask ---------------------------------------------------------------------
class AskRequest(BaseModel):
    # Required, no default: a question sent to the wrong corpus would fail silently ("not found").
    corpus: CorpusName
    question: str = Field(min_length=1, max_length=2000)
    # Names of the documents to search in, inside that corpus (metadata filter on "source").
    sources: list[str] = Field(min_length=1)


class ChunkRef(BaseModel):
    label: str          # "C1", "C2"... as shown to the LLM and the user
    id: str             # chunk id in the vector DB: "{source}::{chunk_index}"
    source: str
    start: int          # character offsets in the source file
    end: int
    start_line: int
    end_line: int
    distance: float     # vector distance: lower = closer. NOT a confidence.
    text: str


AskStatus = Literal[
    "answered",         # answer with at least one verified citation
    "not_found",        # the LLM says the excerpts do not contain the answer
    "unverified",       # the LLM claims an answer but cites nothing that was retrieved
    "invalid_output",   # the LLM output did not match the schema, even after a retry
    "provider_error",   # rate limit, timeout, network or server error from the LLM provider
]


class AskResponse(BaseModel):
    status: AskStatus
    message: str        # human-readable explanation of the status, shown in the UI
    answer: str | None = None
    citations: list[ChunkRef] = []
    retrieved: list[ChunkRef] = []
    warnings: list[str] = []


# --- Documents ---------------------------------------------------------------
class DocumentInfo(BaseModel):
    name: str
    size_bytes: int
    # Number of chunks of this document in the collection of the current settings (0 = not indexed).
    chunks: int
