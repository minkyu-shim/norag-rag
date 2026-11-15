import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app import llm, rag, main
from app.schemas import ChunkRef

# End to end, WITHOUT the network and without an index: fake retrievers and a fake LLM provider.
# What is checked: the answer carries a trace_id, the trace shows every step, and the provider
# only ever saw the redacted question while the user gets the original data back.


def chunk(i, text="Refunds are paid within 14 days."):
    return ChunkRef(label=f"C{i}", id=f"billing.md::{i}", source="billing.md", start=0, end=10,
                    start_line=1, end_line=3, text=text)


class FakeProvider:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, response_format):
        self.prompts.append(messages[0]["content"])
        content = self.replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(rag, "check_ready", lambda *a, **k: None)
    monkeypatch.setitem(rag.RETRIEVERS, "dense", lambda c, q, s, st, n: [chunk(1), chunk(2)])
    monkeypatch.setitem(rag.RETRIEVERS, "bm25", lambda c, q, s, st, n: [chunk(2), chunk(3)])
    monkeypatch.setattr(main, "load_settings", lambda: SimpleNamespace(chunk_size=800, overlap_percent=10, top_k=3))
    return TestClient(main.app)     # no "with": the startup (indexing) is not run


def ask(client, question):
    return client.post("/api/ask", json={"corpus": "kestrel", "question": question, "sources": ["billing.md"]})


def test_answer_has_a_trace_with_every_step(client, monkeypatch):
    provider = FakeProvider([json.dumps({"answer": "14 days.", "found": True, "citations": ["C1"]})])
    monkeypatch.setattr(llm, "client", provider)
    r = ask(client, "How long do refunds take?")
    assert r.status_code == 200 and r.json()["status"] == "answered"
    trace = client.get(f"/api/traces/{r.json()['trace_id']}").json()
    names = [s["name"] for s in trace["spans"]]
    assert names == ["check_ready", "retrieve.dense", "retrieve.bm25", "fusion.rrf", "prompt.build",
                     "llm.call", "llm.validate", "verify.citations"]
    assert trace["attrs"]["outcome"] == "answered"
    call = trace["spans"][5]["attrs"]
    assert call["prompt_tokens"] == 100 and call["completion_tokens"] == 20
    assert trace["attrs"]["total_tokens"] == 120
    # chunk ids, never chunk texts
    assert "Refunds are paid" not in json.dumps(trace)
    assert trace["spans"][3]["attrs"]["chunks"][0]["found_by"]


def test_personal_data_never_reaches_the_provider_or_the_trace_but_comes_back_to_the_user(client, monkeypatch):
    provider = FakeProvider([json.dumps({"answer": "I will write to [EMAIL_1].", "found": True, "citations": ["C1"]})])
    monkeypatch.setattr(llm, "client", provider)
    r = ask(client, "Write to jane.doe@example.com about my refund, my name is Jane Doe")
    assert "jane.doe@example.com" not in provider.prompts[0] and "Jane Doe" not in provider.prompts[0]
    assert "[EMAIL_1]" in provider.prompts[0]
    assert r.json()["answer"] == "I will write to jane.doe@example.com."
    trace = client.get(f"/api/traces/{r.json()['trace_id']}").json()
    assert "jane.doe@example.com" not in json.dumps(trace) and "Jane Doe" not in json.dumps(trace)
    assert trace["attrs"]["redactions"]["EMAIL"] >= 1


def test_invalid_output_shows_two_llm_calls_in_the_trace(client, monkeypatch):
    monkeypatch.setattr(llm, "client", FakeProvider(["not json", "still not json"]))
    r = ask(client, "q?")
    assert r.status_code == 502 and r.json()["status"] == "invalid_output"
    trace = client.get(f"/api/traces/{r.json()['trace_id']}").json()
    assert [s["name"] for s in trace["spans"]].count("llm.call") == 2
    assert trace["status"] == "error" and trace["attrs"]["outcome"] == "invalid_output"


def test_unknown_trace_is_a_404(client):
    assert client.get("/api/traces/nope").status_code == 404
