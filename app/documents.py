import shutil
from pathlib import Path
from app.config import CORPORA_DIR, CORPORA, ALLOWED_EXTENSIONS
from app.chunking import chunk_text
from app import store

# Document management: each corpus is simply a folder data/corpora/<corpus>/, one file per document.
# The file name is the document id inside its corpus (it is stored as "source" in the chunk metadata,
# which is what lets the user choose which documents a question is asked against).
# Every function takes the corpus first: there is no "current corpus" hidden on the server.

# Chroma refuses too large batches in one add(): we send the chunks by groups.
ADD_BATCH_SIZE = 1000


class DocumentError(Exception):
    """A problem with a document that the user caused and can fix (wrong type, empty file...)."""


def corpus_dir(corpus):
    # The corpus name ends up in a path and in a collection name: only the known ones are accepted.
    if corpus not in CORPORA:
        raise DocumentError(f"Unknown corpus {corpus!r}. Known: {', '.join(CORPORA)}")
    return CORPORA_DIR / corpus


def seed_corpus(corpus):
    # First start: the corpus folder is empty, so we copy its seed files in.
    folder = corpus_dir(corpus)
    folder.mkdir(parents=True, exist_ok=True)
    if not any(folder.iterdir()):
        for seed in CORPORA[corpus]:
            shutil.copy(seed, folder / seed.name)
            print(f"Seeded corpus {corpus!r} with {seed.name}")


def seed_all():
    for corpus in CORPORA:
        seed_corpus(corpus)


def list_document_paths(corpus):
    return sorted(p for p in corpus_dir(corpus).iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS)


def read_document(corpus, name):
    return (corpus_dir(corpus) / name).read_text(encoding="utf-8")


def check_name(name):
    # Only a plain file name with an allowed extension: no "../" tricks, no PDF, etc.
    if not name or Path(name).name != name or name.startswith("."):
        raise DocumentError(f"Invalid file name: {name!r}")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentError(f"Unsupported file type for {name!r}. Allowed: {allowed}")


def save_document(corpus, name, content_bytes):
    check_name(name)
    path = corpus_dir(corpus) / name
    if path.exists():
        raise DocumentError(f"A document named {name!r} already exists in {corpus!r}. Delete it first.")
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise DocumentError(f"{name!r} is not UTF-8 text.")
    if not text.strip():
        raise DocumentError(f"{name!r} is empty.")

    path.write_text(text, encoding="utf-8")


def delete_document(corpus, name):
    check_name(name)
    path = corpus_dir(corpus) / name
    if not path.exists():
        raise DocumentError(f"No document named {name!r} in {corpus!r}.")
    path.unlink()
    # Remove its chunks from EVERY collection of the corpus, not only the current settings' one,
    # otherwise a deleted document would come back when switching settings.
    for collection in store.all_collections(corpus):
        collection.delete(where={"source": name})


# --- Indexing phase: document -> chunks -> embeddings -> vector DB ------------
def index_document(corpus, name, settings):
    collection = store.get_collection(corpus, settings)

    # Re-indexing a document: first remove its old chunks, so we never keep duplicates.
    collection.delete(where={"source": name})

    chunks = chunk_text(read_document(corpus, name), settings.chunk_size, settings.overlap_percent)

    # Chroma computes the embeddings itself (with store.embedding_function) when we add documents.
    for batch_start in range(0, len(chunks), ADD_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + ADD_BATCH_SIZE]
        collection.add(
            ids=[f"{name}::{batch_start + i}" for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            # The metadata is what makes a reference verifiable: which file, which characters, which lines.
            metadatas=[
                {
                    "source": name,
                    "chunk_index": batch_start + i,
                    "start": c["start"],
                    "end": c["end"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                }
                for i, c in enumerate(batch)
            ],
        )
    print(f"Indexed {corpus}/{name}: {len(chunks)} chunks into {collection.name}")
    return len(chunks)


def index_all(corpus, settings):
    return {path.name: index_document(corpus, path.name, settings) for path in list_document_paths(corpus)}


def index_missing(corpus, settings, names=None):
    # Index only the documents that have no chunks yet for these settings (all documents by default).
    # Shared by the app startup and eval/run_eval.py, so both follow the same rule.
    if names is None:
        names = [path.name for path in list_document_paths(corpus)]
    for name in names:
        if not is_indexed(corpus, name, settings):
            index_document(corpus, name, settings)


def is_indexed(corpus, name, settings):
    return store.has_chunks(store.get_collection(corpus, settings), name)
