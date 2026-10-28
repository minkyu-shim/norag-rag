import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.config import CHROMA_DIR, EMBEDDING_MODEL

# The vector DB: ChromaDB, stored on disk in data/chroma/ (same library as ask_rag_google.py).
#
# One collection per chunking setting. The collection name says how its chunks were made:
#   chunk_size=800, overlap=10%  ->  "chunks_800_10"
#
# Why? Chunk size and overlap change the chunks themselves, so the old embeddings are useless
# for the new setting: everything must be re-indexed. Keeping one collection per setting means
#   - we never mix chunks made with different settings by mistake,
#   - switching back to a setting already indexed costs nothing,
#   - top_k only changes the query, so it never needs a re-index.
# The embedding model is NOT in the name: changing it means deleting data/chroma/ (see config.py).

# The embedding model runs locally (downloaded on the first run). Loaded once, used for
# both indexing and querying, so both always use the SAME model.
embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection_name(chunk_size, overlap_percent):
    return f"chunks_{chunk_size}_{overlap_percent}"


def get_collection(chunk_size, overlap_percent):
    return chroma_client.get_or_create_collection(
        name=collection_name(chunk_size, overlap_percent),
        embedding_function=embedding_function,
        # Cosine distance: 0 = same direction, 2 = opposite. Easier to read than the default (L2).
        metadata={"hnsw:space": "cosine"},
    )


def all_collections():
    # Used when deleting a document: its chunks must disappear from EVERY setting's collection.
    # (Depending on the chromadb version, list_collections() returns names or Collection objects.)
    names = [getattr(c, "name", c) for c in chroma_client.list_collections()]
    return [chroma_client.get_collection(name=n, embedding_function=embedding_function) for n in names]


def count_chunks(collection, source):
    return len(collection.get(where={"source": source}, include=[])["ids"])
