import re
import sqlite3
from contextlib import closing
from app.config import BM25_FILE
from app.schemas import ChunkRef
from app import store

# V3 — The keyword retriever: BM25, stored in SQLite FTS5 (data/bm25.sqlite).
#
# Why BM25 next to the vectors? Dense retrieval matches MEANING, so two error codes that mean almost
# the same thing (KST-2040 "timed out" vs KST-2044 "refused") get almost the same embedding.
# BM25 matches the WORDS: "kst-2044" is a rare token, so the chunk that contains it wins.
#
# Why SQLite FTS5 (and not rank_bm25 / bm25s)?
#   - it ships with Python (no new dependency, no server), and the index lives on disk
#   - it handles add/delete row by row, exactly like a Chroma collection: no full rebuild on upload
#   - its bm25() ranking function is the standard Okapi BM25 (k1 = 1.2, b = 0.75)
#
# Same organisation as the vector DB (see store.py): ONE TABLE PER (corpus, chunk setting), named like
# the Chroma collection ("kestrel_chunks_800_10"), with the SAME chunk ids ("{source}::{index}").
# Same ids in both stores is what lets RRF (rag.py) add up the ranks of one chunk found by both.

# --- Tokenization ------------------------------------------------------------
# Done in Python, the same way for the chunks and for the question:
#   - lowercase
#   - an identifier stays ONE token: "KST-2044" -> "kst-2044", "203.0.113.10" -> "203.0.113.10"
#     and we ALSO add its parts ("kst", "2044"), so "error 2044" still matches
#   - a short stopword list ("the", "what"...): with OR queries they would match every chunk
#   - no stemming: dense retrieval already covers "refund"/"refunds", and a stemmer could cut identifiers
# The dots of a sentence end are not kept: the pattern needs a character AFTER the "." or "-".
TOKEN_PATTERN = re.compile(r"\w+(?:[-.]\w+)*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "should", "so", "that", "the",
    "their", "them", "there", "they", "this", "to", "was", "we", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
}

# FTS5 splits the text again with its own tokenizer. "tokenchars '-._'" tells it that these characters
# are part of a token, so it splits exactly where tokenize() did (on spaces), and "kst-2044" survives.
FTS_TOKENIZER = "unicode61 tokenchars '-._'"


def tokenize(text):
    tokens = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        word = match.group()
        parts = re.split(r"[-.]", word)
        candidates = [word] + parts if len(parts) > 1 else [word]
        tokens.extend(t for t in candidates if t not in STOPWORDS)
    return tokens


# --- Storage -----------------------------------------------------------------
def connect():
    # A new connection for every call: FastAPI runs the routes in several threads,
    # and a SQLite connection must not be shared between threads.
    BM25_FILE.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(BM25_FILE)


def table_name(corpus, settings):
    # The same rule as the vector DB, so the two stores can never disagree on "which setting".
    return store.collection_name(corpus, settings)


def ensure_table(conn, table):
    # Only "tokens" is searched; the other columns are carried along (UNINDEXED) to build the ChunkRef.
    # (char_start/char_end, not start/end: END is an SQL keyword.)
    conn.execute(
        f'CREATE VIRTUAL TABLE IF NOT EXISTS "{table}" USING fts5('
        f"tokens, chunk_id UNINDEXED, source UNINDEXED, char_start UNINDEXED, char_end UNINDEXED, "
        f"start_line UNINDEXED, end_line UNINDEXED, text UNINDEXED, "
        f'tokenize="{FTS_TOKENIZER}")'
    )


def all_tables(conn, corpus):
    # FTS5 also creates "shadow" tables (x_data, x_idx...): only the virtual tables are ours.
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
    ).fetchall()
    return [name for (name,) in rows if name.startswith(store.collection_prefix(corpus))]


# --- Updates -----------------------------------------------------------------
def index_chunks(corpus, settings, name, ids, chunks):
    # Called by documents.index_document with the SAME ids and chunks as the vector DB.
    table = table_name(corpus, settings)
    with closing(connect()) as conn, conn:      # "with conn": one transaction, committed at the end
        ensure_table(conn, table)
        # Re-indexing a document: first remove its old rows, so we never keep duplicates.
        conn.execute(f'DELETE FROM "{table}" WHERE source = ?', (name,))
        conn.executemany(
            f'INSERT INTO "{table}" (tokens, chunk_id, source, char_start, char_end, start_line, end_line, text) '
            f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(" ".join(tokenize(c["text"])), chunk_id, name, c["start"], c["end"],
              c["start_line"], c["end_line"], c["text"])
             for chunk_id, c in zip(ids, chunks)],
        )


def delete_source(corpus, name):
    # Like the vector DB: a deleted document disappears from EVERY setting's table of its corpus.
    with closing(connect()) as conn, conn:
        for table in all_tables(conn, corpus):
            conn.execute(f'DELETE FROM "{table}" WHERE source = ?', (name,))


def has_chunks(corpus, settings, name):
    table = table_name(corpus, settings)
    with closing(connect()) as conn, conn:
        ensure_table(conn, table)
        return conn.execute(f'SELECT 1 FROM "{table}" WHERE source = ? LIMIT 1', (name,)).fetchone() is not None


# --- Retrieval ---------------------------------------------------------------
def retrieve(corpus, question, sources, settings, n):
    # Unique tokens of the question, each one in double quotes (so FTS5 reads it as a plain word,
    # not as its query syntax), joined with OR: a chunk only needs ONE of the words to be a candidate,
    # and bm25() ranks higher the chunks with more, and rarer, question words.
    tokens = list(dict.fromkeys(tokenize(question)))
    if not tokens:
        return []           # the question was only stopwords: BM25 has nothing to search for
    match = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)

    table = table_name(corpus, settings)
    placeholders = ", ".join("?" for _ in sources)
    with closing(connect()) as conn, conn:
        ensure_table(conn, table)
        # bm25() is NEGATIVE, and lower = better: ORDER BY ascending gives the best chunk first.
        rows = conn.execute(
            f'SELECT chunk_id, source, char_start, char_end, start_line, end_line, text, bm25("{table}") AS score '
            f'FROM "{table}" WHERE "{table}" MATCH ? AND source IN ({placeholders}) '
            f"ORDER BY score LIMIT ?",
            (match, *sources, n),
        ).fetchall()

    return [
        ChunkRef(
            label=f"C{i + 1}",
            id=chunk_id, source=source, start=start, end=end, start_line=start_line, end_line=end_line,
            # Shown as a positive number: higher = better match. Like the distance, NOT a confidence.
            bm25_score=round(-score, 4),
            text=text,
        )
        for i, (chunk_id, source, start, end, start_line, end_line, text, score) in enumerate(rows)
    ]
