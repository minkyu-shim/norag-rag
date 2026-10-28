import json
from pydantic import ValidationError
from app.config import (
    SETTINGS_FILE, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_PERCENT, DEFAULT_TOP_K,
)
from app.schemas import Settings

# The settings changed from the UI are saved in data/settings.json,
# so they survive a restart of the server. No need to edit the code to change them.


def default_settings():
    return Settings(
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap_percent=DEFAULT_OVERLAP_PERCENT,
        top_k=DEFAULT_TOP_K,
    )


def load_settings():
    if not SETTINGS_FILE.exists():
        return default_settings()
    try:
        return Settings.model_validate_json(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (ValidationError, ValueError):
        # A broken settings file must not break the app: fall back to the defaults.
        print(f"WARNING: {SETTINGS_FILE} is invalid, using default settings.")
        return default_settings()


def save_settings(settings):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings.model_dump(), indent=2), encoding="utf-8")
