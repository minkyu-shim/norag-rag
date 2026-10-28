import os
from pathlib import Path
from dotenv import load_dotenv

# All settings of the app live here, as module-level constants (same idea as in ask_rag_google.py).
# Settings that the user can change from the UI (chunk size, overlap, top-k) only have their
# DEFAULT value here: the current value is stored in data/settings.json (see app/settings.py).

load_dotenv()

# --- Paths -------------------------------------------------------------------
# Every path is built from the repo root, so the app works whatever the current directory is.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # the corpus: one .md/.txt file per document
CHROMA_DIR = DATA_DIR / "chroma"      # the vector DB (separate from ./chroma_db used by the scripts)
SETTINGS_FILE = DATA_DIR / "settings.json"
WEB_DIR = ROOT_DIR / "web"

# On first start, data/docs/ is empty: we copy this file in, so the app has a corpus to work with.
SEED_DOCUMENT = ROOT_DIR / "docs" / "CrimeAndPunishment.txt"

# --- Documents ---------------------------------------------------------------
ALLOWED_EXTENSIONS = {".md", ".txt"}

# --- Embeddings --------------------------------------------------------------
# Same local model as ask_rag_google.py (384 dimensions, ~256 word-pieces of input).
# WARNING: indexing and querying MUST use the same model. If you change it,
# delete data/chroma/ so that everything gets re-indexed with the new model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Chunking / retrieval defaults (lab settings) ----------------------------
DEFAULT_CHUNK_SIZE = 800        # characters per chunk
DEFAULT_OVERLAP_PERCENT = 10    # step = 800 * (1 - 0.10) = 720 characters
DEFAULT_TOP_K = 5
MIN_CHUNK_LENGTH = 20           # drop tiny leftover chunks (e.g. the end of a file)

# --- LLM ---------------------------------------------------------------------
# Gemini, through Google's OpenAI-compatible endpoint (same as the scripts).
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# gemini-2.5-flash-lite (used by the scripts) is still listed but answers 404 since Oct. 2026.
# We pin an exact version, not "gemini-flash-lite-latest", so eval numbers stay comparable over time.
MODEL = "gemini-3.1-flash-lite"
LLM_TIMEOUT_SECONDS = 30
# How many extra attempts when the LLM answer does not match the schema.
LLM_INVALID_OUTPUT_RETRIES = 1
