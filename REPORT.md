# REPORT — RAG app (V1 → V3)

> TODO (team): the decision log must be **written by you**. The entries below hold the facts
> (options, choice, numbers measured on Oct. 28th for V1, Nov. 14th for V2/V3). Rewrite every "Why"
> in your own words, check the grades (sections 5.3 and 6.4), and delete this note.
>
> Sections 1–4: V1. Section 5: V2 (two corpora, evaluation, Words tab). Section 6: V3 (BM25, concept graph, fusion).

## 1. Team details

| Member | Worked on |
|---|---|
| TODO | TODO |
| TODO | TODO |

## 2. Decision log

### Corpus — *Crime and Punishment* + the Denis needle
- **Options:** Crime and Punishment (already in the repo); several Gutenberg books; our own documents.
- **Picked:** Crime and Punishment (1.18 MB, ~1600 chunks at 800 chars).
- **Why:** TODO. Facts: it is large enough that retrieval matters. The LLM already knows the book, so most questions *could* be answered without retrieval. The injected Denis Amselemovitch passage (line ~14613) is the only content the LLM cannot know.

### Stack — FastAPI + plain HTML/JS
- **Options:** FastAPI + vanilla JS; FastAPI + React (Vite); Streamlit.
- **Picked:** FastAPI (backend, Pydantic built in) + one static page, no build step.
- **Why:** TODO. Facts: the spec asks for the pipeline "behind a web API". Streamlit has no separate API layer.

### Embedding model — `all-MiniLM-L6-v2` (local)
- **Options:** MiniLM (local, 384 dims); a bigger local model; an API embedding model (Gemini).
- **Picked:** MiniLM, same as the reference script.
- **Why:** TODO. Facts: no API cost or quota, and indexing the whole book takes ~20 s on a laptop CPU. The input limit is 256 word-pieces (~1000 chars), so long chunks get truncated. Changing the model means deleting `data/chroma/`.

### Vector DB — ChromaDB, one collection per (chunk size, overlap)
- **Options:** one collection, wiped on every settings change; one collection per setting.
- **Picked:** one per setting (`chunks_800_10`, `chunks_500_20`, ...).
- **Why:** TODO. Facts: chunks from different settings are never mixed, switching back to an already indexed setting is free, and top-k needs no re-index. Cost: disk space, and a delete has to clean every collection.

### Chunking — fixed 800-char windows, 10 % overlap
- **Options:** fixed windows (lab); paragraphs (reference script); sentence-aware.
- **Picked:** fixed windows, the lab setting. **Naive on purpose:** V1 is the baseline.
- **Why:** TODO. Every chunk stores `start`/`end` offsets and line numbers, so a reference can be checked in the file.

### LLM output — JSON mode + Pydantic `LLMAnswer {answer, found, citations}`
- **Options:** free text; JSON mode + validation; native structured output.
- **Picked:** `response_format=json_object`, then `LLMAnswer.model_validate_json`. One retry, then status `invalid_output`.
- **Why:** TODO. `found` makes "I don't know" a field we can test rather than a sentence to parse.

### Citations — short labels `C1..Ck`, checked on the server
- Excerpts are labelled `C1..Ck` in the prompt. A citation that is not one of the retrieved labels is dropped with a warning, and if nothing valid remains the status is `unverified`.
- **Why:** TODO (answers "how do you know a cited reference was actually retrieved?").

### Controlled failures — one explicit status per outcome
`answered | not_found | unverified | invalid_output | provider_error`, plus HTTP 400 for "unknown document / not indexed" and 422 for invalid input. The retrieved chunks are always returned. No retries on 429 (`max_retries=0`): the user is told right away.

### Evaluation — evidence quotes, not chunk ids
- 25 questions in `eval/questions.json`: 10 easy, 8 hard (paraphrase, no character name, vocabulary mismatch), 2 needle, 5 unanswerable.
- The ground truth is a quote, located in the file as a character span. A hit means a retrieved chunk covers ≥ 50 % of the span, so the measure works for any chunk size.
- **Why / how many questions is enough:** TODO.

## 3. Baseline numbers (retrieval only, `python eval/run_eval.py`)

| Setting | n | hit@1 | hit@3 | hit@5 | MRR@5 |
|---|---|---|---|---|---|
| chunk 800, overlap 10 % | 20 | 0.15 | 0.25 | 0.25 | 0.20 |
| chunk 500, overlap 20 % | 20 | 0.10 | 0.20 | 0.25 | 0.16 |

Mean top-1 cosine distance (800/10): answerable **0.450**, unanswerable **0.481**. These are almost the same, so a distance threshold alone would not separate "nothing relevant" from a real hit.

## 4. Known failures

| # | Question | What we observed | Likely cause |
|---|---|---|---|
| 1 | *What is the full name of the main character?* (e01) | The evidence chunk is not even in the top 200 | The question shares no rare word with the passage; MiniLM matches on topic, not on "name" |
| 2 | *Which character might be a time traveller who used AI to get into the story?* (n02) | The needle chunk ranks 8th (outside the top 5), while *Who is Denis Amselemovitch?* ranks 1st | Paraphrase/vocabulary mismatch: the query says "time traveller", the text says "artifact from the future" |
| 3 | *What document did the drunk official's daughter have to carry…?* (h08) | The evidence ("yellow ticket") ranks 151st | The hard questions avoid character names, and without names MiniLM retrieval falls apart (hard hit@5 = 0.12) |
| 4 | *What exactly does the hero say when he gives himself up?* (h05) | The confession is not in the top 200 | A single query, and many chunks talk about confession and the murder; the key sentence is diluted in an 800-char window |
| 5 | *What is the capital of Australia?* (u02) | 5 chunks are still retrieved and sent to the LLM | Top-k is always returned, with no relevance threshold (weak spot #4) |
| 6 | Needle passage | Split across two chunks (`::1024` = lines 14600–14614, `::1025` = lines 14613–14631) | Fixed windows ignore paragraph boundaries |
| 7 | LLM end-to-end (`run_eval.py --llm`, V2) | Not run in V1 (no API key set when the baseline was measured). Run in V2 on Kestrel: see section 5.3 | |

---

## 5. V2 — Two corpora, evaluation, Words tab

### 5.1 Question sets and the no-context test
- `eval/questions_kestrel.jsonl`: 12 questions (the 10 starter questions + q11, q12 added in V3). Types: exact_identifier 2, split_answer 2, vocabulary 1, versions 2, two_hop 3, not_in_corpus 2.
- `eval/questions_crime.jsonl`: 25 questions (the V1 set in JSONL): 20 answerable (10 `other`, 9 `vocabulary`, 1 `exact_identifier`), 5 `not_in_corpus`.
- No-context test (`eval/no_context.py`, the topic is given, no excerpt): results and marks in `eval/no_context_<corpus>.json` (`known_without_rag`).

| Corpus | Answerable answered correctly with NO retrieval | Unanswerable handled correctly |
|---|---|---|
| Kestrel | **0 / 10** (9 "I don't know", q02 invents a cause) | 1 / 2 (q09 asserts "no iPhone app", q10 says it does not know) |
| Crime and Punishment | **17 / 20** (all but e03, n01, n02) | 4 / 5 (u02: answers "Canberra") |

- Kestrel questions test the RAG. Most Crime questions do **not**: the model knows the book. Only the two needle questions (n01, n02) and e03 (it says "cigarette case", the book says "silver watch") need retrieval.
- **Decision (replace or keep the Crime questions):** TODO (team).

### 5.2 Results table (retrieval, chunk 800 / overlap 10 %, top-k 5 unless written)
Kestrel, n = 10 answerable (`not_in_corpus` has no evidence, so no hit@k):

| Configuration | hit@1 | hit@5 | recall@5 | recall@10 | exact_id hit@5 | split recall@5 | two_hop recall@5 | versions hit@5 | vocabulary hit@5 |
|---|---|---|---|---|---|---|---|---|---|
| dense c500/o20 | 0.20 | 0.80 | 0.72 | 0.88 | 0.50 | 0.88 | 0.50 | 1.00 | 1.00 |
| dense c800/o10 | 0.70 | 0.90 | 0.80 | 0.90 | 0.50 | 1.00 | 0.67 | 1.00 | 1.00 |
| dense c1200/o10 | 0.90 | 1.00 | 0.90 | 0.95 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 |
| bm25 | 0.70 | 0.90 | 0.68 | 0.72 | 1.00 | 0.38 | 0.67 | 1.00 | **0.00** |
| dense + bm25 (RRF) | 0.60 | **1.00** | 0.82 | 0.90 | 1.00 | 0.62 | 0.67 | 1.00 | 1.00 |
| dense + bm25 + graph | 0.70 | **1.00** | 0.82 | **0.95** | 1.00 | 0.62 | 0.67 | 1.00 | 1.00 |
| graph alone | 0.20 | 0.20 | 0.20 | 0.20 | 0.50 | 0.00 | 0.33 | 0.00 | 0.00 |

Crime and Punishment, n = 20 answerable:

| Configuration | hit@1 | hit@5 | recall@10 | other hit@5 | vocabulary hit@5 | needle (exact_id) hit@5 |
|---|---|---|---|---|---|---|
| dense c500/o20 | 0.10 | 0.25 | 0.35 | 0.40 | 0.00 | 1.00 |
| dense c800/o10 | 0.15 | 0.25 | **0.50** | 0.30 | 0.11 | 1.00 |
| dense c1200/o10 | 0.05 | 0.25 | 0.30 | 0.30 | 0.11 | 1.00 |
| bm25 | 0.20 | 0.30 | 0.45 | 0.30 | 0.22 | 1.00 |
| dense + bm25 (RRF) | **0.25** | **0.35** | 0.40 | 0.40 | 0.22 | 1.00 |

Files: `eval/results/<corpus>_<retrievers>_c<size>_o<overlap>_20261114-*.json` (crime c500 / c1200: `crime_c500_o20_20261103-102727.json`, `crime_c1200_o10_20261103-102752.json`). Tables made with `python eval/compare.py <files> --metrics hit@1,hit@5,recall@5,recall@10`.

**What we changed, and what moved:**
- **Chunk size (dense only).** Kestrel: 500 → 800 → 1200 moves hit@1 from 0.20 to 0.70 to 0.90 and recall@5 from 0.72 to 0.80 to 0.90. The Kestrel sections are short (an error code, a service): a bigger window keeps the heading ("### KST-2044") and its text in one chunk. Crime: hit@5 stays at 0.25 for all three sizes; recall@10 is best at 800 (0.50).
- **Adding BM25 (RRF).** Kestrel hit@5 0.90 → 1.00 (q01: KST-2044 goes from rank 7 to rank 4; BM25 alone puts it at 1). But hit@1 drops 0.70 → 0.60 and split_answer recall@5 1.00 → 0.62 (see "worse", section 6.3). Crime hit@5 0.25 → 0.35, but recall@10 0.50 → 0.40.
- **Adding the graph.** Same hit@5 / recall@5 as dense + bm25. hit@1 0.60 → 0.70 and recall@10 0.90 → 0.95 (section 6.3).

**Default configuration:** `dense + bm25`, chunk 800 / overlap 10 %, top-k 5 (`DEFAULT_RETRIEVERS` in `app/schemas.py`, `data/settings.json`). Kestrel hit@5 1.00, recall@5 0.82; Crime hit@5 0.35 (best of the measured configurations on both corpora at c800). Not measured: dense + bm25 at c1200, which dense alone suggests could be better on Kestrel.
- **Why this default:** TODO (team).

### 5.3 LLM answers, graded (Kestrel, dense + bm25, c800/o10)
File: `eval/results/kestrel_dense+bm25_c800_o10_llm_20261114-112156.json` (rules in its `grading` key, one `grade` / `failure` / `grading_note` per answer).

Grading rules (written before grading):
- **correct**: every key fact of the expected answer is there, and the cited chunks say it. Unanswerable: the app says it is not in the documents.
- **partial**: what it says is true and supported, but a key fact is missing.
- **wrong**: false for the current documents (e.g. an outdated policy given as current), or an answer to an unanswerable question.
- **unsupported**: says things the cited chunks do not say.
- Every answer that is not correct gets a failure: **retrieval** (a needed chunk was not in the top 5) or **generation** (it was given and not used).

| Grade | Count | Questions |
|---|---|---|
| correct | 8 | q01, q02, q05, q06, q07, q09, q10, q12 |
| partial | 3 | q04 (generation), q08 (retrieval), q11 (retrieval) |
| wrong | 1 | q03 (retrieval) |
| unsupported | 0 | |

- Both `not_in_corpus` questions get `not_found`: no invented answer.
- q03 is the only wrong answer: it gives the **2025** failed-payment rules (retries at 3 / 7 days, 14-day grace) as current. The 2026 chunk with the 21-day grace period ranked 6th.
- q08 and q11 miss the same thing: the **second hop** (the teams table in `service-catalog.md`) was not in the top 5, so the model cannot name the channel or the hours.
- Cost per question (LLM run): 1 LLM call, ~1 243 prompt tokens, ~82 completion tokens, ~1.7 s of LLM latency, ~21 ms of retrieval.

### 5.4 Two misses, diagnosed with the Words tab (question mode)

**Kestrel — q05 "I lost my phone and I can't log in anymore. What can I do?" with BM25 alone (evidence not in the top 20).**
- `phone` and `anymore` are **absent** from the corpus (0 chunks). `log` is in 3 chunks, `lost` in 1: none of them is the evidence chunk.
- The evidence chunk (`product-guide.md::6`) says "no longer have your authenticator device", "backup codes", "reset your second factor": **no word in common** with the question once the stopwords are removed.
- So BM25 cannot find it, by construction. Dense finds it at rank 1 (it matches the meaning). This is the "vocabulary" case of Kestrel.

**Crime — h08 "What document did the drunk official's daughter have to carry because of how she earned money?" (evidence not in the top 20, any retriever).**
- No word is absent, but the only question word in the evidence chunk (`CrimeAndPunishment.txt::44`) is `daughter`, and `daughter` is in 49 chunks.
- The words that would point to the answer are not in the question: the passage says "yellow ticket", the question says "document". `document` (4 chunks), `carry` (17), `earned` (3) are rare, but none is in the evidence chunk.
- Common words carry no signal: `did` is in 26 % of the chunks, `she` in 39 %, `s` in 64 %.
- So BM25 has nothing to match, and dense finds chunks about drunk officials and daughters (Marmeladov's speech is long, many chunks) before this one.

### 5.5 Words tab and evaluation from the UI
- Words tab: corpus mode (vocabulary, word → chunks) and question mode (question → words, with the evidence chunks of a question of the set). It uses the BM25 tokenizer and the BM25 table of the current chunk setting, so it shows exactly the words the keyword retriever sees.
- Evaluation tab: runs `evaluation.evaluate` from the UI (retrievers to use, LLM answers on/off) and saves the same result file as `eval/run_eval.py`.
- **Decisions and why (Words: which tokenizer, per chunk or per document, computed or cached):** TODO (team). Facts: same tokens as BM25, counted per chunk, computed on every request (~0.1 s on Crime and Punishment, 1 604 chunks).

---

## 6. V3 — BM25, concept graph, fusion

### 6.1 BM25
- **Index:** SQLite FTS5 in `data/bm25.sqlite`, one table per (corpus, chunk setting), named like the Chroma collection (`kestrel_chunks_800_10`), with the **same chunk ids**, so RRF can add up the ranks of one chunk found by both. Ranking: FTS5 `bm25()` (Okapi, k1 = 1.2, b = 0.75).
- **Tokenization (in Python, same for chunks and question):** lowercase; an identifier stays one token (`kst-2044`, `203.0.113.10`) **plus** its parts (`kst`, `2044`); a short stopword list; **no stemming**.
- **Updates:** rows are added / deleted chunk by chunk on upload, delete and re-index, like the Chroma collection. No full rebuild.
- **Why:** TODO (team).
- Cost of no stemming, seen in q03: the question says `fails`, the 2026 grace-period chunk says `failed`. `fails` is in 3 chunks, all of them **2025** or the first 2026 chunk, so BM25 never ranks the 21-day chunk (`billing-policy-2026.md::5`).

### 6.2 The concept graph
- **Schema (fixed):** entity types `ErrorCode, Service, Team, Person, Channel, Plan, Feature`; relations `raised_by, escalate_to, owned_by, on_call_channel, lead, includes, replaced`, each with allowed source/target types (`app/graph.py`).
- **Extraction:** one LLM call per chunk (43 chunks), JSON validated with Pydantic, and every triple must come with a **quote found in its chunk** (or it is dropped).
- **Entity resolution:** names normalized (casefold, accents, `#`, markdown and punctuation removed) and merged when type + normalized name match. No embedding / LLM judge (fewer merges, but no wrong merges).
- **Retrieval:** local walk. Question entities are found by string match, then edges are followed both ways up to 3 hops (hubs with degree ≥ 12 are reached, not walked through). Each edge gives the chunks of its quote. The ranked list goes into RRF with dense and BM25.
- **Optional features (teams of 3):** none.

**The graph in numbers (Kestrel, c800/o10, prompt v2):**

| | |
|---|---|
| Entities (nodes) | 32 |
| Relations (edges) | 28: owned_by 7, lead 7, on_call_channel 7, escalate_to 5, raised_by 1, replaced 1, includes 0 |
| Triples | 28 extracted, 28 valid, 0 dropped |
| Merges | 0 between different names; 9 nodes built from several chunks (e.g. `Team:data integrations` from 3 chunks) |
| Build cost | 44 LLM calls, 34 249 prompt + 2 441 completion tokens, 403 s (free tier, 4.5 s between calls). Rebuild from cache: 0 calls |
| Prompt v1 | 45 calls, 24 valid triples, **0 escalate_to** (the prompt did not say how the documents write it) |

**Extraction quality (10 random triples, `eval/graph_triples_graded_kestrel.json`): 9 correct / 1 wrong / 0 unsupported.**

**One bad triple, and what it cost:** `KST-2044 --escalate_to--> Data Integrations`.
- The quote is the "Escalate" line of **KST-2040**, the section just above, in the same chunk. KST-2044 says "Escalate: no".
- Cost: **KST-2040 is not in the graph at all**. On q11 ("KST-2040 … which team, and when can it be reached?") the graph walk has no seed and returns nothing.
- The second hop (the teams table) was not found by dense or BM25 either. Result: q11 recall@5 = 0.5 and the answer is graded partial ("when" is missing).
- Other gaps: `includes` (plan → feature) was never extracted, and `raised_by` only once (KST-5031 → postroom). Most error codes do not name their service, so most chains start at a team.
- **How to evaluate a concept graph, beyond 10 triples:** TODO (team).

### 6.3 What happened to the graph's chunks after fusion
- **q08** (KST-5031 → postroom → Messaging → #oncall-messaging): the graph ranks the teams-table chunk (`service-catalog.md::3`, 2nd hop) **3rd**. Dense ranks it 12th, BM25 not at all. After RRF it is **6th**, just outside the top 5: RRF score 0.0298, against 0.0298 / 0.0318 for chunks that dense and BM25 both rank in their top 13.
  - A chunk found by **one** retriever loses to chunks found by **two**, even ranked low by both.
- **What we changed, measured:** top-k 5 → 10. With k = 10 the chunk is in: two_hop recall@10 = **0.83** with the graph, 0.67 without; overall recall@10 0.90 → 0.95.
  - Not tried: a weight for the graph in RRF, or more candidates.
- **q01**: the graph helps hit@1. dense + bm25 puts KST-2044 at rank 4, dense + bm25 + graph at rank 1. The graph reaches that chunk through the **wrong** triple above, which points into the same chunk.

**One question where adding a retriever made things worse: q03** ("What happens, step by step, when a customer's card payment fails?").
- Dense alone: the four evidence quotes at ranks 2, 1, 1, 1 (recall@5 = 1.00).
- dense + bm25: ranks 2, **6, 6, 6** (recall@5 = 0.25). The 2026 grace-period chunk drops out of the top 5.
- Why: BM25 ranks the **2025** policy chunks high (`fails`, `card`, `payment` are there, no stemming for `failed`) and the 2026 grace-period chunk not at all. With RRF, the 2025 chunks found by both retrievers pass it.
- Effect on the answer: the LLM gets the 2025 rules and gives them as current (graded **wrong**, 5.3). Same pattern on q05, a vocabulary question: dense rank 1, dense + bm25 rank 3, because BM25 finds nothing and only adds noise.

### 6.4 Cost per question, per configuration (retrieval only, Kestrel)
| Configuration | Retrieval ms | LLM calls at question time |
|---|---|---|
| bm25 | 0.7 | 0 |
| graph | 0.4 | 0 |
| dense | 5.4 | 0 |
| dense + bm25 | 5.0 | 0 |
| dense + bm25 + graph | 5.3 | 0 |
| + LLM answer (dense + bm25) | +~1 700 ms | 1 call, ~1 243 + 82 tokens |

The graph costs nothing per question: its cost is at indexing time (44 LLM calls, 403 s).

## 7. Known failures (V2 / V3)

| # | Question | What we observed | Likely cause |
|---|---|---|---|
| 8 | q03, card payment fails (Kestrel) | The LLM gives the 2025 rules as current (wrong answer) | dense + bm25 drops the 2026 grace-period chunk to rank 6; BM25 has no stemming (`fails` / `failed`), and the prompt does not ask the model to prefer the newer document |
| 9 | q08, q11 two_hop (Kestrel) | 2nd hop (teams table) not in the top 5; partial answers | One-retriever chunks lose in RRF (q08); a wrong triple leaves KST-2040 out of the graph (q11) |
| 10 | q11 | Status `not_found` although half of the answer is right | `found` is one boolean: no way to say "partly found" |
| 11 | q05 with BM25 alone | Evidence not in the top 20 | No common word: `phone` and `anymore` are not in the corpus |
| 12 | Crime question set | 17 / 20 answerable questions answered right with no retrieval | The model knows the novel; only the needle questions test the RAG |
| 13 | Triple `KST-2044 escalate_to Data Integrations` | Wrong triple accepted | The quote check only proves the quote is in the chunk, not that it belongs to the source entity (two sections in one chunk) |

---

## 8. V4 Part B — Concept map explorer

### 8.1 What the user gets
- A **Concept map** tab (Kestrel). On the left, the 32 concepts of the graph, grouped by type (team, service, error code, person, channel), with a search box.
- Click a concept to see its neighbourhood drawn as a small map: the concept in the middle, its neighbours around it, arrows in the direction of each relation, relations in plain words ("owned by", "on-call channel"). Click a neighbour to move to it; **Back** returns.
- Under the map, every link as a sentence (*KST-5031 escalate to → postroom*), the quote the build read it from, and the passage(s) it comes from. A click opens the passage with the names highlighted.
- **Documents that mention it**: per document, how many times and in which passages.
- **Ask about this**: opens the Ask tab with a question that fits the type of concept ("What does the Platform team own, and how can support reach it?"), and switches the graph retriever on. The user can edit it before asking.

### 8.2 Decisions (facts; the "why" is TODO (team))
- **Read-only, no LLM call.** The graph is already built (V3). Code: `app/explorer.py`, `web/explorer.js`; API `GET /api/concepts`, `GET /api/concept`.
- **One concept's neighbourhood (1 hop)**, not the whole graph at once.
- **Links point to the chunks of the current chunk setting.** Each edge stores its quote's character span, mapped onto the chunks with `coverage()`, same rule as the graph retriever and the evaluation.
- **Mentions** use the same name matching as the graph retriever (`graph_retrieval.alias_pattern`, shared since V4).
- **Duplicate edges are shown as one link with several sources** (see 8.4).
- **Why:** TODO (team).

### 8.3 Cost, failures, checks
- **Cost per click:** 0 LLM calls, 0 tokens. Server time for one concept, measured over the 32 concepts × 3: p50 0.6 ms, p95 0.8 ms, max 1.3 ms; about 2–6 ms in the browser. The cost of the feature is the graph build, once: 44 LLM calls, ~34 k prompt tokens, 403 s (section 6.2). Every response carries its timing per step (`graph.load`, `graph.neighbours`, `chunks.map`, `documents.mentions`), ids and counts only, never chunk text.
- **Failures, and what the user sees:**

| Case | What the user sees |
|---|---|
| No graph for the corpus (Crime and Punishment) | "There is no concept map for 'crime' yet", and the command that builds it |
| A concept with no link | The concept alone, and "no link in the concept map" |
| A document deleted after the build | The link stays, with "… was deleted after the concept map was built" |
| Unknown concept id | HTTP 404 with a message |
| A name no document contains | "No document contains the name … as such" (see Analytics Eng below) |

- **Checked on 7 concepts:**

| Concept | Links | What we checked / found |
|---|---|---|
| postroom | 4 | KST-5031 (raised by, escalate to), owned by Messaging, replaced mailman: the q08 chain is visible in one view |
| Messaging | 3 | Re-centring from postroom, lead Inès Moreau, #oncall-messaging; Back returns to postroom |
| Platform | 4 | The node the plan called a hub: only 4 links at this size, below `HUB_DEGREE` = 12 |
| Payments Core | 3 | Was 5 before the fix: two links were stored twice (8.4) |
| KST-2044 | 1 | Shows the **wrong** triple *KST-2044 escalate to Data Integrations* with its quote, which is the KST-2040 line: the map makes the error easy to see |
| KST-5031 | 2 | Two relations to the same neighbour: one node, label "escalate to · raised by" |
| Analytics Eng | 1 | A ghost concept (8.4): no document mentions it |

### 8.4 What the explorer showed about the graph
- **Duplicate edges.** The same triple was extracted from two overlapping chunks (`service-catalog.md::3` and `::4`): *Payments Core lead Jonas Berg* and *Payments Core on_call_channel #oncall-payments* are each stored twice. The graph counts 28 edges, but there are fewer distinct links.
- **A missed merge.** `Analytics Eng` (Team) is a separate node from `Analytics Engine`. The LLM copied a name cut at a chunk boundary (the KST-3012 "Escalate" line). No document contains "Analytics Eng" on its own, and the normalization rule (same type + same normalized name) cannot merge it.
- **The wrong KST-2044 triple** (section 6.2) is visible as soon as one clicks KST-2044.

## 9. V4 — Clean-up for a first-time user, and the install path

### 9.1 Clean-up
- **Hidden by default, shown with "Show technical details":** the retriever switches, the scores and RRF ranks of each passage, the graph path, the cost line, the technical part of a passage's details, and the Words, Evaluation and Settings tabs. The choice is remembered in the browser.
- **A first-time user sees** three tabs: Ask, Concept map, Documents.
- **Plain words:**
  - Header: "Ask the documents" instead of "Hybrid RAG".
  - Document sets: "Kestrel Analytics (help centre)", "Crime and Punishment (novel)" instead of `kestrel` / `crime`.
  - Status titles: "Answer found in your documents", "Not in your documents", "The AI service did not answer" instead of `answered` / `not_found` / `provider_error`. The server's message follows.
  - Index warning: "These documents are not ready for questions yet … Prepare them now".
  - Buttons and headings: "Add the document", "Passages the answer is based on".
- **Loading states:** "Searching the documents…" while a question runs; "Loading…" in the concept map.
- **Small screens:** the tab bar wraps instead of widening the page.
- **Why:** TODO (team).

### 9.2 Install path
- **Problem found:** `data/` is not in git, so a fresh clone had no concept graph (and building one costs ~45 LLM calls).
- **Fix:** the graph is committed as a seed file (`docs/graphs/kestrel.json`, 16 KB). It is copied to `data/graph/` on first start, like the seed documents (`graph.seed_graphs()`), and never overwrites a rebuilt graph.
- **README:** rewritten so the first section is a quick start (clone → venv → install → `.env` → `uvicorn` → open the page), with what the first start does and a troubleshooting list. The NoRAG / RAG scripts of the start of the course moved to the end.
- **Tested:** fresh clone of the branch into an empty folder, new venv, `pip install -r requirements.txt`, no `.env`, `uvicorn app.main:app`.
  - Both corpora were seeded and indexed with no manual step, and the concept map had 32 concepts.
  - Ask without a key returned "The LLM is not configured: set GOOGLE_API_KEY in .env and restart", together with the 5 retrieved passages.
  - Not covered: the embedding model was already in the machine's Hugging Face cache, so the first download was not timed; only Python 3.14 was tried.
- **Why:** TODO (team).
