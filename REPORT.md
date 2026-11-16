# REPORT.md TODOs

One short answer per TODO, in REPORT.md order.

## 1. Team details
- **Minkyu SHIM:** V1 app, V2 (corpora, evaluation, Words tab), V3 (BM25, graph, fusion), V4 Part B (concept map), clean-up and install path.
- **Armand:** V4 Part A (traces), Part C (personal data redaction), tests.

## 2. Decision log — Why
| Decision | Why |
|---|---|
| Corpus: Crime and Punishment | Big enough (1 600 chunks) that top-k matters. The needle is the one fact the model can't know, so it shows whether retrieval worked. |
| Stack: FastAPI + plain JS | The spec wants a real web API. FastAPI ships with Pydantic, and plain JS needs no build step. Streamlit has no API layer. |
| Embeddings: MiniLM, local | Free, no quota, ~20 s to index the book. It's the reference model, so V1 is a clean baseline. Accepted cost: chunks over ~1 000 chars get cut off. |
| One Chroma collection per setting | Chunks from different settings never mix, and switching back to an indexed setting is free. Disk space is a cheap price. |
| Fixed 800-char windows | Naive on purpose: V1 is the baseline. Offsets on every chunk make each reference checkable. |
| JSON mode + Pydantic `LLMAnswer` | Bad output becomes a handled status, not a crash. `found` lets us test "I don't know" as a field. |
| Citations `C1..Ck` checked on the server | Labels are a closed set. The server drops any citation that wasn't retrieved, so a made-up reference can't reach the user. |
| Evaluation: evidence quotes + how many questions | Quote spans survive chunk-size changes; chunk ids don't. 20 answerable questions is the minimum to see big moves (±0.05 = 1 question). More per type before trusting small differences. |

## 5. V2
- **5.1 Keep or replace the Crime questions?** Keep them, and say so. They measure retrieval (hit@k) even though the model knows the answers. Only n01, n02 and e03 test the RAG end to end, so grade answers on Kestrel instead.
- **5.2 Why dense + bm25 at c800/o10, k5?** Best hit@5 on both corpora (Kestrel 1.00, Crime 0.35). BM25 catches exact ids, dense catches paraphrases. Known cost: q03.
- **5.5 Words tab choices.** Same tokenizer as BM25, so it shows what the keyword retriever sees. Per chunk, because retrieval works per chunk. Computed on every request: 0.1 s, nothing to keep in sync.

## 6. V3
- **6.1 Why BM25 in SQLite FTS5?** Built into Python, no new service. Same chunk ids as Chroma, so RRF can merge the two lists. Custom tokens keep `kst-2044` whole. No stemming keeps it simple (cost: `fails` vs `failed`).
- **6.2 How to evaluate the graph beyond 10 triples?**
  - Precision: grade a bigger random sample.
  - Recall: hand-list the triples in a few chunks and count how many were found.
  - Usefulness: two_hop recall with and without the graph.

## 8–11. V4 — Why
- **8.2 Concept map:** reuses the graph we already built, at zero LLM cost per click. One hop stays readable. Every link shows its quote, so the user can check it, and bad triples became visible.
- **9.1 Clean-up:** a first-time user needs three tabs and plain words. Course tools and scores stay one checkbox away.
- **9.2 Install path:** `data/` isn't in git. Committing the 16 KB graph avoids 45 LLM calls after a clone.
- **10.2 Traces:** answers why an answer went wrong (which step, how slow, how many tokens). Local SQLite needs no service, and nothing leaves the machine. Scrubbing happens in one place, so no caller can leak keys or documents.
- **11.2 Redaction:** personal data shouldn't reach the provider or the logs. Rules, not an LLM, because an LLM detector would send the data out to find it. Few false alarms beat catching everything.
