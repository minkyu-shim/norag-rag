import pytest
from app import tracing


@pytest.fixture(autouse=True)
def traces_in_tmp(tmp_path, monkeypatch):
    # Every test writes its traces to its own temporary database, never to data/traces.sqlite.
    monkeypatch.setattr(tracing, "TRACES_FILE", tmp_path / "traces.sqlite")
