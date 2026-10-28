import shutil
from pathlib import Path
from app.config import DOCS_DIR, SEED_DOCUMENT, ALLOWED_EXTENSIONS
from app.chunking import chunk_text
from app import store

# Document management: the corpus is simply the folder data/docs/, one file per document.
# The file name is the document id (it is stored as "source" in the chunk metadata,
# which is what lets the user choose which documents a question is asked against).

# Chroma refuses too large batches in one add(): we send the chunks by groups.
ADD_BATCH_SIZE = 1000


class DocumentError(Exception):
    """A problem with a document that the user caused and can fix (wrong type, empty file...)."""


def seed_corpus():
    # First start: data/docs/ is empty, so we copy the book in.
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    if not any(DOCS_DIR.iterdir()) and SEED_DOCUMENT.exists():
        shutil.copy(SEED_DOCUMENT, DOCS_DIR / SEED_DOCUMENT.name)
        print(f"Seeded the corpus with {SEED_DOCUMENT.name}")


def list_document_paths():
    return sorted(p for p in DOCS_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS)


def read_document(name):
    return (DOCS_DIR / name).read_text(encoding="utf-8")


def check_name(name):
    # Only a plain file name with an allowed extension: no "../" tricks, no PDF, etc.
    if not name or Path(name).name != name or name.startswith("."):
        raise DocumentError(f"Invalid file name: {name!r}")
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DocumentError(f"Unsupported file type for {name!r}. Allowed: {allowed}")


def save_document(name, content_bytes):
    check_name(name)
    if (DOCS_DIR / name).exists():
        raise DocumentError(f"A document named {name!r} already exists. Delete it first.")
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise DocumentError(f"{name!r} is not UTF-8 text.")
    if not text.strip():
        raise DocumentError(f"{name!r} is empty.")

    (DOCS_DIR / name).write_text(text, encoding="utf-8")


def delete_document(name):
    check_name(name)
    path = DOCS_DIR / name
    if not path.exists():
        raise DocumentError(f"No document named {name!r}.")
    path.unlink()
    # Remove its chunks from EVERY collection, not only the current settings' one,
    # otherwise a deleted document would come back when switching settings.
    for collection in store.all_collections():
        collection.delete(where={"source": name})


# --- Indexing phase: document -> chunks -> embeddings -> vector DB ------------
def index_document(name, settings):
    collection = store.get_collection(settings.chunk_size, settings.overlap_percent)

    # Re-indexing a document: first remove its old chunks, so we never keep duplicates.
    collection.delete(where={"source": name})

    chunks = chunk_text(read_document(name), settings.chunk_size, settings.overlap_percent)

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
    print(f"Indexed {name}: {len(chunks)} chunks into {collection.name}")
    return len(chunks)


def index_all(settings):
    return {path.name: index_document(path.name, settings) for path in list_document_paths()}


def is_indexed(name, settings):
    collection = store.get_collection(settings.chunk_size, settings.overlap_percent)
    return store.count_chunks(collection, name) > 0
