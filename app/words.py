import sqlite3
from collections import Counter
from contextlib import closing
from types import SimpleNamespace
from app import bm25, documents, evaluation
from app.chunking import find_span, coverage

# V2 — Words analytics: what WORDS does the corpus contain, and where?
#
# Why words, in a RAG built on embeddings? When dense retrieval misses, looking at the words often tells
# why: the question uses a word the corpus never uses ("phone", "locked out"), or a word that is in
# hundreds of chunks, so it cannot point to one. That diagnosis is what this module serves.
#
# Decisions (see REPORT.md):
#   - A "word" is a token of the BM25 tokenizer (bm25.tokenize): lowercase, identifiers kept whole
#     ("kst-2044") plus their parts ("kst", "2044"), stopwords dropped, no stemming.
#     Same tokens as BM25, so the Words tab shows exactly what the keyword retriever sees.
#   - Counted per CHUNK of the current setting (chunk size / overlap), read from the BM25 table,
#     which already holds the tokens of every chunk. No second index to keep in sync.
#     Consequence: with overlap, a word in the overlapping part is counted in both chunks.
#   - Computed on every request, nothing cached: Crime and Punishment (~1650 chunks) takes ~0.1 s.


class WordsNotReadyError(Exception):
    """The corpus is not indexed for the current setting: nothing to count."""


def load_chunks(corpus, settings):
    # Every chunk of the corpus for this setting, with its tokens already split.
    table = bm25.table_name(corpus, settings)
    with closing(bm25.connect()) as conn:
        try:
            rows = conn.execute(
                f'SELECT chunk_id, source, char_start, char_end, start_line, end_line, tokens, text FROM "{table}"'
            ).fetchall()
        except sqlite3.OperationalError:      # no such table: never indexed with this setting
            rows = []
    if not rows:
        raise WordsNotReadyError(
            f"Corpus {corpus!r} is not indexed for chunk size {settings.chunk_size} / "
            f"overlap {settings.overlap_percent}%. Click 'Re-index'.")
    return [
        SimpleNamespace(id=chunk_id, source=source, start=start, end=end, start_line=start_line,
                        end_line=end_line, counts=Counter(tokens.split()), text=text)
        for chunk_id, source, start, end, start_line, end_line, tokens, text in rows
    ]


def chunk_info(chunk, word=None):
    info = {"id": chunk.id, "source": chunk.source, "start": chunk.start, "end": chunk.end,
            "start_line": chunk.start_line, "end_line": chunk.end_line, "text": chunk.text}
    if word is not None:
        info["occurrences"] = chunk.counts[word]
    return info


# --- Corpus mode: vocabulary ----------------------------------------------------
def vocabulary(corpus, settings, contains="", order="frequent", limit=200):
    chunks = load_chunks(corpus, settings)
    total = Counter()           # word -> occurrences in the whole corpus
    in_chunks = Counter()       # word -> number of chunks that contain it
    for c in chunks:
        total.update(c.counts)
        in_chunks.update(c.counts.keys())

    words = [w for w in total if contains.lower() in w]
    if order == "rare":
        # Rarest first: the words that appear in a single chunk are the ones BM25 can pin down.
        words.sort(key=lambda w: (in_chunks[w], total[w], w))
    elif order == "alpha":
        words.sort()
    else:
        words.sort(key=lambda w: (-total[w], w))

    return {
        "corpus": corpus,
        "chunks": len(chunks),
        "tokens": sum(total.values()),
        "distinct_words": len(total),
        "matching_words": len(words),
        "words": [{"word": w, "count": total[w], "chunks": in_chunks[w]} for w in words[:limit]],
    }


# --- Corpus mode: word -> chunks ---------------------------------------------------
def normalize(word):
    # The user may type "KST-2044" or "Refunds,": apply the same tokenizer as the index.
    # Several tokens (e.g. "locked out") -> we keep them all; the caller decides what to do.
    return bm25.tokenize(word)


def word_chunks(corpus, settings, word):
    tokens = normalize(word)
    if not tokens:
        return {"word": word, "ignored": True, "count": 0, "chunk_count": 0, "chunks": [],
                "message": f"{word!r} is a stopword (or has no letters/digits): it is not indexed."}
    # A single word is expected; for "KST-2044" the tokenizer also returns its parts, the first token is the whole.
    token = tokens[0]
    chunks = load_chunks(corpus, settings)
    found = sorted((c for c in chunks if c.counts[token]), key=lambda c: (-c.counts[token], c.source, c.start))
    return {
        "word": token,
        "ignored": False,
        "count": sum(c.counts[token] for c in found),
        "chunk_count": len(found),
        "total_chunks": len(chunks),
        "chunks": [chunk_info(c, token) for c in found],
    }


# --- Question mode: question -> words --------------------------------------------
def question_set(corpus):
    # The question sets of the evaluation, for the picker of the Words tab.
    return [{"id": q["id"], "type": q["type"], "answerable": q["answerable"], "question": q["question"]}
            for q in evaluation.load_questions(corpus)]


def evidence_chunks(corpus, chunks, question_id):
    # The chunks that hold the answer: same rule as the evaluation (chunk covers >= 50% of a quote).
    q = next((q for q in evaluation.load_questions(corpus) if q["id"] == question_id), None)
    if q is None:
        return None, []
    ids = []
    for ev in q["evidence"]:
        try:
            span = find_span(documents.read_document(corpus, ev["source"]), ev["quote"])
        except (ValueError, FileNotFoundError):
            continue        # the document was deleted, or the quote is not in it anymore
        ids += [c.id for c in chunks
                if c.source == ev["source"] and coverage(c, span) >= evaluation.MIN_COVERAGE and c.id not in ids]
    return q, ids


def question_words(corpus, settings, question, question_id=None, max_chunks=20):
    chunks = load_chunks(corpus, settings)
    q, evidence_ids = evidence_chunks(corpus, chunks, question_id) if question_id else (None, [])
    if q is not None:
        question = q["question"]
    evidence = [c for c in chunks if c.id in evidence_ids]

    # Every word of the question, in order, with the stopwords marked (they are not indexed at all).
    raw = bm25.TOKEN_PATTERN.findall(question.lower())
    stopwords = [w for w in dict.fromkeys(raw) if w in bm25.STOPWORDS]
    words = []
    for token in dict.fromkeys(bm25.tokenize(question)):
        found = sorted((c for c in chunks if c.counts[token]), key=lambda c: -c.counts[token])
        words.append({
            "word": token,
            "count": sum(c.counts[token] for c in found),
            "chunk_count": len(found),
            # Share of the corpus' chunks that contain the word: near 0 = specific, high = says nothing.
            "chunk_share": round(len(found) / len(chunks), 4),
            # Does the word appear in the chunk(s) that hold the answer? (only for a question of the set)
            "in_evidence": [c.id for c in evidence if c.counts[token]],
            "chunks": [{"id": c.id, "source": c.source, "occurrences": c.counts[token]} for c in found[:max_chunks]],
        })

    return {
        "question": question,
        "question_id": q["id"] if q else None,
        "type": q["type"] if q else None,
        "answerable": q["answerable"] if q else None,
        "total_chunks": len(chunks),
        "stopwords": stopwords,
        "absent": [w["word"] for w in words if w["chunk_count"] == 0],
        "words": words,
        "evidence_chunks": [chunk_info(c) for c in evidence],
    }
