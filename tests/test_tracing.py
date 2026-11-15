import pytest
from app import tracing


def test_one_trace_with_nested_spans():
    with tracing.trace("ask", corpus="kestrel", question="hi") as t:
        with tracing.span("retrieve.dense", n=20) as s:
            s.set(chunk_ids=["a.md::0", "a.md::1"])
            with tracing.span("child"):
                pass
    saved = tracing.get_trace(t.id)
    assert saved["name"] == "ask" and saved["status"] == "ok"
    names = [s["name"] for s in saved["spans"]]
    assert names == ["retrieve.dense", "child"]
    dense, child = saved["spans"]
    assert child["parent_id"] == dense["id"]
    assert dense["attrs"]["chunk_ids"] == ["a.md::0", "a.md::1"]
    assert dense["duration_ms"] >= 0


def test_no_active_trace_means_no_op():
    with tracing.span("x") as s:        # must not raise, must not store anything
        s.set(a=1)
    assert tracing.list_traces() == []


def test_nested_trace_becomes_a_span():
    with tracing.trace("index.corpus") as t:
        with tracing.trace("index.document", document="a.md"):
            pass
    saved = tracing.get_trace(t.id)
    assert [s["name"] for s in saved["spans"]] == ["index.document"]
    assert len(tracing.list_traces()) == 1


def test_an_exception_marks_the_trace_and_is_not_swallowed():
    with pytest.raises(ValueError):
        with tracing.trace("ask") as t:
            with tracing.span("boom"):
                raise ValueError("bad")
    saved = tracing.get_trace(t.id)
    assert saved["status"] == "error"
    assert saved["spans"][0]["status"] == "error" and "bad" in saved["spans"][0]["attrs"]["error"]


def test_no_api_key_no_whole_document_no_personal_data_in_a_trace():
    key = "AIza" + "x" * 35
    with tracing.trace("ask", question="my name is Jane Doe, mail jane@example.com, phone +33 6 12 34 56 78") as t:
        with tracing.span("llm.call", api_key="should-not-be-stored", note=f"using {key}",
                          document_text="lorem ipsum " * 500, role="privacy@kestrel.example") as s:
            s.set(prompt_tokens=10, completion_tokens=5)
    saved = tracing.get_trace(t.id)
    dump = str(saved)
    assert key not in dump and "should-not-be-stored" not in dump
    assert "jane@example.com" not in dump and "Jane Doe" not in dump and "6 12 34 56 78" not in dump
    assert "privacy@kestrel.example" in dump                       # a role address is not personal data
    span = saved["spans"][0]
    assert span["attrs"]["api_key"] == "[SECRET REMOVED]"
    assert len(span["attrs"]["document_text"]) < 700               # cut
    assert span["attrs"]["prompt_tokens"] == 10                    # "tokens" are not secrets
    assert saved["attrs"]["redactions"] == {"PERSON": 1, "EMAIL": 1, "PHONE": 1}
    assert saved["attrs"]["total_tokens"] == 15


def test_retention_deletes_old_traces():
    with tracing.trace("old") as t:
        pass
    import sqlite3
    conn = tracing.connect()
    conn.execute("UPDATE traces SET started_at = '2020-01-01T00:00:00' WHERE id = ?", (t.id,))
    conn.commit(); conn.close()
    with tracing.trace("recent"):
        pass
    assert tracing.purge_old_traces(days=30) == 1
    assert [x["name"] for x in tracing.list_traces()] == ["recent"]
