import contextvars
import json
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from app.config import TRACES_FILE, TRACE_RETENTION_DAYS, MAX_TRACE_STRING
from app import redaction

# V4 Part A — Observability. One TRACE per request (a question, an upload, a re-index, a graph build),
# one SPAN per step inside it (a retriever, the fusion, the prompt, one LLM call, the validation...).
#
#   from app import tracing
#   with tracing.trace("ask", corpus="kestrel") as t:          # one trace per request
#       with tracing.span("retrieve.dense", n=20) as s:        # one span per step
#           ...
#           s.set(chunk_ids=ids)                               # ids only, never document text
#   t.id    # goes back to the UI, so the "Details" button can open it
#
# Storage: SQLite (data/traces.sqlite), written ONCE when the trace ends (one transaction, so a crash
# never leaves half a trace). No server, no account: the traces never leave this machine.
#
# What can NEVER end up in a trace (enforced in clean(), the single door to the database, not by
# the good will of each caller):
#   - API keys (key-shaped strings are replaced),
#   - whole documents (every string is cut at MAX_TRACE_STRING characters: pass chunk ids instead),
#   - personal data: e-mails, phone numbers, names go through app/redaction.py first; the trace
#     only keeps HOW MANY were replaced (trace.attrs.redactions).
#
# Outside a trace (the evaluation, the tests), trace/span do nothing: the code can call them anywhere.

_current_trace = contextvars.ContextVar("current_trace", default=None)
_current_span = contextvars.ContextVar("current_span", default=None)
_db_lock = threading.Lock()

KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),         # Google API keys
    re.compile(r"\bsk-[0-9A-Za-z_\-]{20,}"),        # OpenAI-style keys
    re.compile(r"\b(?:Bearer|Token)\s+[0-9A-Za-z._\-]{20,}", re.I),
]
SECRET_KEY_NAMES = re.compile(r"(api[_-]?key|secret|password|token(?!s)|authorization)", re.I)


# --- Storage -----------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, started_at TEXT NOT NULL,
    duration_ms REAL, status TEXT NOT NULL, attrs TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, parent_id TEXT, name TEXT NOT NULL,
    offset_ms REAL NOT NULL, duration_ms REAL, status TEXT NOT NULL, attrs TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS traces_started ON traces(started_at);
"""


def connect():
    TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TRACES_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")     # the app and a terminal script can write at the same time
    conn.executescript(SCHEMA)
    return conn


def purge_old_traces(days=TRACE_RETENTION_DAYS):
    """Retention: traces older than `days` days are deleted. Called once at startup."""
    limit = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _db_lock, closing(connect()) as conn, conn:
        old = [r["id"] for r in conn.execute("SELECT id FROM traces WHERE started_at < ?", (limit,))]
        for trace_id in old:
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace_id,))
            conn.execute("DELETE FROM traces WHERE id = ?", (trace_id,))
    return len(old)


# --- The single door: everything that is stored goes through clean() ----------------
def clean(value, counts, key=""):
    """Makes a value safe to store: no key, no long text, no personal data. Counts the redactions."""
    if SECRET_KEY_NAMES.search(key or ""):
        return "[SECRET REMOVED]"
    if isinstance(value, str):
        for pattern in KEY_PATTERNS:
            value = pattern.sub("[KEY REMOVED]", value)
        result = redaction.redact(value)
        for kind, n in result.counts.items():
            counts[kind] = counts.get(kind, 0) + n
        text = result.text
        if len(text) > MAX_TRACE_STRING:
            text = text[:MAX_TRACE_STRING] + f"… [cut, {len(text)} characters]"
        return text
    if isinstance(value, dict):
        return {str(k): clean(v, counts, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v, counts, key) for v in list(value)[:200]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return clean(str(value), counts, key)


# --- Spans and traces -----------------------------------------------------------------
class Span:
    def __init__(self, name, trace, parent, attrs):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.trace = trace
        self.parent_id = parent.id if parent else None
        self.attrs = dict(attrs)
        self.status = "ok"
        self.t0 = time.perf_counter()
        self.offset_ms = round((self.t0 - trace.t0) * 1000, 1) if trace else 0
        self.duration_ms = None

    def set(self, **attrs):
        self.attrs.update(attrs)
        return self

    def error(self, message):
        self.status = "error"
        self.attrs["error"] = str(message)
        return self


class Trace(Span):
    def __init__(self, name, attrs):
        self.spans = []
        self.t0 = time.perf_counter()
        super().__init__(name, self, None, attrs)
        self.offset_ms = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")

    @property
    def total_tokens(self):
        return sum(s.attrs.get("prompt_tokens", 0) + s.attrs.get("completion_tokens", 0)
                   for s in self.spans if isinstance(s.attrs.get("prompt_tokens", 0), int))


class _NoSpan:
    """What trace()/span() give when no trace is active: accepts everything, stores nothing."""
    id = None
    status = "ok"

    def set(self, **attrs):
        return self

    def error(self, message):
        return self


@contextmanager
def trace(name, **attrs):
    """One trace per request. Inside an active trace it becomes a plain span (a re-index of ALL documents
    is one trace; each document in it is a span)."""
    if _current_trace.get() is not None:
        with span(name, **attrs) as s:
            yield s
        return

    t = Trace(name, attrs)
    token_t, token_s = _current_trace.set(t), _current_span.set(t)
    try:
        yield t
    except BaseException as e:
        t.error(f"{type(e).__name__}: {e}")
        raise
    finally:
        t.duration_ms = round((time.perf_counter() - t.t0) * 1000, 1)
        _current_trace.reset(token_t)
        _current_span.reset(token_s)
        save(t)


@contextmanager
def span(name, **attrs):
    t = _current_trace.get()
    if t is None:
        yield _NoSpan()
        return
    s = Span(name, t, _current_span.get(), attrs)
    t.spans.append(s)
    token = _current_span.set(s)
    try:
        yield s
    except BaseException as e:
        s.error(f"{type(e).__name__}: {e}")
        raise
    finally:
        s.duration_ms = round((time.perf_counter() - s.t0) * 1000, 1)
        _current_span.reset(token)


def current_trace_id():
    t = _current_trace.get()
    return t.id if t else None


def save(t):
    counts = {}
    attrs = clean(t.attrs, counts)
    spans = [(s.id, t.id, s.parent_id, s.name, s.offset_ms, s.duration_ms, s.status,
              json.dumps(clean(s.attrs, counts), ensure_ascii=False)) for s in t.spans]
    attrs["redactions"] = counts
    attrs["total_tokens"] = t.total_tokens
    try:
        with _db_lock, closing(connect()) as conn, conn:
            conn.execute("INSERT INTO traces VALUES (?,?,?,?,?,?)",
                         (t.id, t.name, t.started_at, t.duration_ms, t.status, json.dumps(attrs, ensure_ascii=False)))
            conn.executemany("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)", spans)
    except sqlite3.Error as e:
        # Observability must never break the app: a trace we cannot write is lost, the request still works.
        print(f"WARNING: could not write trace {t.id}: {e}")


# --- Reading (for GET /api/traces) -------------------------------------------------------
def get_trace(trace_id):
    with closing(connect()) as conn:
        row = conn.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        spans = conn.execute("SELECT * FROM spans WHERE trace_id = ? ORDER BY offset_ms, rowid", (trace_id,)).fetchall()
    return {**dict(row), "attrs": json.loads(row["attrs"]),
            "spans": [{**dict(s), "attrs": json.loads(s["attrs"])} for s in spans]}


def list_traces(limit=50):
    with closing(connect()) as conn:
        rows = conn.execute("SELECT * FROM traces ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
    return [{**dict(r), "attrs": json.loads(r["attrs"])} for r in rows]
