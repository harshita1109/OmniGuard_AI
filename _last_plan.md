# SilentGuard AI — Updated Rebuild Plan (Hybrid RAG, Resume-Ready)

Here's the full plan, now with RAG integrated as a core feature (not an afterthought).

---

## The architecture you're building

```
Client → FastAPI (Uvicorn) → HybridDetector
                                  │
                       ┌──────────┴──────────┐
                       │   Vector path       │
                       │   (fast, cheap)     │
                       │   Embedder + kNN    │
                       └──────────┬──────────┘
                                  │ score
                            ┌─────▼─────┐
                            │  Router   │
                            └─────┬─────┘
                                  │ uncertain?
                       ┌──────────▼──────────┐
                       │    RAG path         │
                       │  Retrieve top-5 +   │
                       │  LLM judges         │
                       └──────────┬──────────┘
                                  │
                          PostgreSQL audit log
                          Redis cache + rate limit
```

The **router** is the architectural keystone — cheap path handles obvious cases, expensive RAG path only kicks in for borderline ones. This is your headline resume story.

---

## Guiding rules (re-read at every phase)

1. **No AI writes your code.** AI explains; you type.
2. **No mystery code** — can't explain a line = can't keep it.
3. **Small frequent commits** in your voice.
4. **Two logbooks**: `NOTES.md` (what you learned), `METRICS.md` (numbers — resume gold).
5. **Old project = reference only.** Never copy-paste.

---

## PHASE 0 — Setup (Day 1)

**Why first**: workflow before substance. Get git/venv/hygiene right and every later phase is smoother.

1. **Create GitHub repo** `silentguard` — public, with README + Python `.gitignore` + MIT license.
2. **Clone locally** to a *fresh* folder (NOT inside the old project).
3. **Create venv** — `python -m venv venv` then activate. (Isolates this project's libs from your system.)
4. **Create starter files** — `README.md` (3 sentences in your voice), `NOTES.md` (Day 1 entry), `METRICS.md` (empty), `requirements.txt`, `.env.example`.
5. **First commit + push** — `"phase 0: project scaffolding"`.

**Verify**: repo visible on GitHub, venv activates, first commit pushed.

---

## PHASE 1 — Core Idea in 30 Lines (Days 2-3)

**Why now**: until you can compare meaning between two sentences in one script, nothing else matters.

**Concepts first**: embedding (text → vector), cosine similarity (angle between meanings), 384 dimensions = MiniLM's output size.

1. **Install** `sentence-transformers`, `numpy`.
2. **`experiment.py`** — load `all-MiniLM-L6-v2`, embed 3 sentences, compute pairwise cosine similarity by hand with numpy. Verify similar sentences are closer.
3. **`detector_v1.py`** — 5 hardcoded normal sentences → mean vector ("centroid") → new input → distance → `normal` or `anomaly`.
4. **Tune threshold** by hand on a few test inputs.

**Verify**: obvious-weird input prints `anomaly`, sensible one prints `normal`, you can explain every line.

---

## PHASE 2 — Real Module (Day 4)

**Why now**: throwaway scripts explore; classes build.

1. Package layout: `src/silentguard/{embedder,detector}.py`, `tests/`, `scripts/`.
2. Extract `Embedder` class (lazy-load model on first use).
3. Extract `Detector` class with `add_baseline()` and `check()`.
4. First pytest: assert an obvious anomaly gets flagged.

**Verify**: `pytest` passes, code organized as a package.

---

## PHASE 3 — Your Own Vector Store (Days 5-7)

**Why now**: builds the *retrieval foundation* both for distance-based detection AND for RAG later.

1. **`VectorStore`** class — `add`, `add_batch`, `search(query, k=5)`, `save`, `load`. Internally a list of numpy arrays + metadata dicts.
2. **Refactor `Detector`** to use kNN search instead of centroid — mean distance to top-k neighbors = anomaly score.
3. **Tests** — add+search, save+load roundtrip, empty store.

**Verify**: save store → restart Python → load → search returns expected results.

---

## PHASE 4 — Real Dataset + Evaluation (Days 8-9)

**Why now**: your **first resume number** comes from this phase. You can't tune what you can't measure.

1. **`scripts/generate_data.py`** — ~300 examples, half normal LLM outputs, half clearly wrong. Save to `data/raw/logs.jsonl`.
2. **`scripts/evaluate.py`** — train/test split, evaluate at thresholds 0.1→0.9, print precision/recall/F1/confusion matrix.
3. **Record in METRICS.md**: `Phase 4 — Vector-only: F1 = 0.86 at threshold 0.42`.

**Verify**: F1 ≥ 0.8 on synthetic data.

---

## PHASE 5 — FastAPI: Your First Backend (Days 10-13)

**Why now**: the biggest learning leap. Take your time.

**Concepts first**: HTTP (request/response, methods, status codes), JSON, Uvicorn (the server), Pydantic (validation).

1. **Install** `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`.
2. **`api.py`** — `POST /check` accepts `{text}`, returns `{is_anomaly, score}`. Define `CheckRequest`/`CheckResponse` as Pydantic models.
3. **`config.py`** — `pydantic-settings` reading from `.env`.
4. **Load detector once at startup** using FastAPI's `lifespan`.
5. **Run** `uvicorn silentguard.api:app --reload`, open `/docs`, try the endpoint.
6. **Write `tests/test_api.py`** with FastAPI's `TestClient`.

**Verify**: `curl POST /check` returns valid JSON, `/docs` loads, tests pass.

---

## PHASE 6 — RAG Layer: LLM-as-Judge (Days 14-17) ⭐

**Why now**: vector detection is fast but imperfect on borderline cases. RAG retrieves similar past examples and asks an LLM to judge. This is the **defining feature** of the project.

**Concepts first**: RAG = retrieval + augmented prompt + generation. LLMs don't know your baseline — retrieved context gives them ground truth.

1. **Pick LLM provider** — OpenAI `gpt-4o-mini` (cheap, easy) OR local Ollama (`llama3.1:8b`, free, slower). Document choice in NOTES.md.
2. **`llm_client.py`** — `LLMClient.judge(new_text, reference_examples)` returning `{is_anomaly, confidence, reason}`.
3. **`prompts/judge.txt`** — prompt template that lists reference examples and asks for JSON verdict.
4. **`rag_judge.py`** — `RAGJudge.judge(text)`:
   1. Embed text.
   2. `vector_store.search(top_k=5)`.
   3. Extract reference texts from retrieved metadata.
   4. Call `llm_client.judge(...)`.
   5. Return verdict.
5. **Test** on 20 hand-picked borderline examples from Phase 4 that the vector detector got wrong.

**Verify**: RAG judge correctly classifies most borderline cases the vector path missed.

---

## PHASE 7 — The Router (Days 18-19) ⭐

**Why now**: combining the two paths is the architectural keystone — and your headline resume number.

1. **Define confidence zones**: `score < 0.2` → normal (fast). `score > 0.6` → anomaly (fast). `0.2 ≤ score ≤ 0.6` → call RAG.
2. **`router.py`** — `HybridDetector.check(text)` returns `{is_anomaly, score, path: "vector"|"rag", reason}`.
3. **Re-run evaluation** — compute hybrid F1 AND % of requests routed to RAG.
4. **Record in METRICS.md**: `Phase 7 — Hybrid: F1 = 0.93, only 18% of requests hit LLM (82% cost saving vs naive RAG)`.
5. **Update API** to return `path` and `reason`.

**This is the bullet that lands the interview.**

---

## PHASE 8 — Logging Cleanup (Day 20)

Replace `print()` with `logging`. Configure once. Log startup, every request (with path taken), every anomaly. Quick phase but essential.

---

## PHASE 9 — WebSockets: Real-Time Monitoring (Days 21-23)

**Why now**: REST is one-shot; monitoring is continuous streams.

**Concepts first**: persistent connections, `async`/`await`, concurrency vs parallelism.

1. **`/ws/monitor`** — accepts `{text}` messages, routes through `HybridDetector`, sends back verdict.
2. **`ConnectionManager`** — dict of `{client_id: WebSocket}`, methods `send_to` and `broadcast`.
3. **Anomaly broadcast** — when any client triggers an anomaly, all connected clients receive an alert.
4. **`scripts/ws_client.py`** — test client.
5. **Concurrency test** — run 3 clients at once, all work independently.

**Verify**: two concurrent clients both work and both receive broadcasts.

---

## PHASE 10 — Docker (Days 24-25)

**Why now**: "works on my machine" is the oldest backend joke. Containerize before adding databases — simpler first.

1. **Dockerfile** — base `python:3.11-slim`, install deps, copy source, expose 8000, CMD uvicorn.
2. **Build + run** — `docker build -t silentguard .` then `docker run -p 8000:8000 silentguard`. Hit from host browser. *The magic moment.*
3. **docker-compose.yml** — just the `api` service for now.

**Verify**: app runs identically via `docker-compose up`.

---

## PHASE 11 — PostgreSQL: Audit Log (Days 26-29)

**Why now**: every check should be persisted for analysis. Real DB earns its place when there's a real reason.

**Concepts first**: relational schemas, ORM, migrations, dependency injection, connection pooling.

1. **Add Postgres to docker-compose**.
2. **SQLModel** (FastAPI's author's library — easier than raw SQLAlchemy).
3. **`Check` table** — id, text, score, is_anomaly, path, reason, client_id, timestamp.
4. **`db.py`** — engine + `get_db()` dependency.
5. **Wire `/check`** to write every check to DB. Add `GET /history?client_id=...`.
6. **Alembic** migrations — initialize, generate first migration, apply. Now schema is version-controlled.

**Verify**: restart DB container → history preserved.

---

## PHASE 12 — Redis: Caching + Rate Limiting (Days 30-31)

**Why now**: re-embedding the same text wastes compute; LLM calls cost money; APIs need abuse protection.

1. **Add Redis to docker-compose**.
2. **Embedding cache** — `sha256(text) → vector`, TTL 24h. Skip the model on cache hit.
3. **RAG verdict cache** — same text + same retrieval → same verdict. TTL 1h.
4. **Rate limit** — `slowapi` middleware, 60 req/min per IP.
5. **Measure** — time 1000 repeated calls. Record in METRICS.md: `15× speedup, 40% LLM call reduction`.

---

## PHASE 13 — Tests, CI, Quality (Days 32-33)

1. **GitHub Actions** `.github/workflows/ci.yml` — ruff (lint), mypy (types), pytest (with coverage) on every PR.
2. **Pre-commit hooks** — same checks locally.
3. **Coverage target** ~70% via `pytest-cov`.
4. **One integration test** that spins up app + DB (testcontainers or compose).

**Verify**: green check on a PR, coverage ≥ 70%.

---

## PHASE 14 — Polish + Portfolio (Days 34-36)

**Why now**: a project no one can run is invisible.

1. **Rewrite README** — what it does, architecture diagram, quickstart (`docker-compose up`), `/docs` screenshot, METRICS section, tech stack.
2. **2-minute screencast demo** — spin up, hit API, show both paths, show DB, show broadcast. Embed in README.
3. **Deploy** (optional, huge ROI) — Railway/Render/Fly.io. A live URL on your resume converts 3-5× better.
4. **"What I learned" section** — 5-10 honest bullets. Your real differentiator.
5. **Final NOTES.md + METRICS.md cleanup** — these become your interview cheat sheet.
6. **Tag** `v1.0`.

---

## Final folder structure

```
silentguard/
├── README.md, NOTES.md, METRICS.md
├── pyproject.toml, requirements.txt, .env.example
├── Dockerfile, docker-compose.yml
├── alembic.ini, alembic/
├── .github/workflows/ci.yml
├── prompts/judge.txt
├── src/silentguard/
│   ├── api.py, config.py, logging.py
│   ├── embedder.py, vector_store.py, detector.py
│   ├── llm_client.py, rag_judge.py, router.py
│   ├── ws_manager.py
│   ├── db.py, models.py, cache.py
├── scripts/{play,generate_data,evaluate,ws_client}.py
├── tests/test_*.py
└── data/raw/logs.jsonl
```

---

## Your target resume bullet

> **SilentGuard AI** — Real-time monitoring system for LLM silent failures
> *Python, FastAPI, PostgreSQL, Redis, Docker, sentence-transformers, OpenAI API*
> - Designed a hybrid anomaly detection pipeline: fast vector-similarity for high-confidence cases, RAG-based LLM judge for uncertain ones — improving F1 from 0.86 to 0.93 while routing only 18% of requests to the expensive LLM path
> - Built a custom vector store with kNN search + persistence, then layered a retrieval pipeline grounding LLM judgments in the top-5 most similar baseline examples
> - Implemented real-time monitoring via WebSockets with concurrent client support and anomaly broadcast
> - Persisted ~10k checks/day to PostgreSQL with Redis-backed embedding + verdict caching (15× repeat-query speedup)
> - Containerized with Docker Compose, CI via GitHub Actions, 70% test coverage

---

## How I'll help (and won't)

**Will**: explain concepts deeply, review code you wrote (no rewrites), quiz you, debug *with* you by asking questions.
**Won't**: write code into your project, hand you snippets to paste, move you forward when you can't explain the current step.

---

## Your very first action — RIGHT NOW

**Phase 0, Step 1**: Go to github.com → New repository → name `silentguard` → Public → check README, Python `.gitignore`, MIT license → Create.

Come back here when it's done and we'll do step 0.2 (clone + venv) together.