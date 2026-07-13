# PRD — Rehnuma: University Admissions & Scholarship Advisor

> **Purpose of this document:** This is the founding context file for the project. If you are an LLM/coding assistant reading this in a fresh session, this document tells you (1) who the developer is and how to work with him, (2) what we are building and why, (3) the exact architecture and stack, and (4) the milestone plan. Treat it as the source of truth.

---

## 1. Developer Context (read this first)

**Who:** Bilal Tahir — university student (4th semester completed) in Pakistan. Strong background in **full-stack web development** (has real experience; understands routing, REST, auth, databases, git). Currently transitioning into **AI engineering** with the goal of mastering **agentic AI / multi-agent systems**. This project is his first full-stack AI project and is meant to be a serious CV/GitHub portfolio piece — not a toy.

**What he already knows well (do NOT over-explain these):**
- **LangChain (modern API):** chat models via `init_chat_model`, tools, messages, structured output, streaming/batch, `create_agent`, middleware (summarization, HITL).
- **LangGraph:** `StateGraph`, state with custom reducers, conditional edges, parallel fan-out, cyclic graphs (writer→reviewer critic loops), `ToolNode`, `interrupt`/`Command` human-in-the-loop, checkpointers, stream modes, **subgraphs**, **supervisor architecture** with `Command(goto=...)` handoffs.
- **RAG:** recursive/semantic chunking, Chroma & FAISS, hybrid retrieval (BM25 + vector ensemble), multi-query retriever, parent-document retriever, contextual compression, Anthropic-style contextual retrieval, citing sources.
- General Python, and web concepts from his full-stack background.

**What he knows in theory but has NOT coded yet (introduce hands-on, briefly explain when first used):**
- Production persistence: `PostgresSaver` checkpointer, LangGraph `Store` (long-term/cross-thread memory), pgvector.
- Reranking (cross-encoder rerankers).
- RAG/agent evaluation (RAGAS, LangSmith evals). Has basic LangSmith tracing setup only.
- Late chunking, agentic RAG, multimodal RAG (conceptual knowledge only).

**New to him (explain as it comes up, but leverage his web background — he'll get it fast):**
- FastAPI (first time; he knows Express-style backends, so map concepts). Python `async/await`, SSE/token streaming to a client.
- Docker / docker-compose (assume beginner unless he says otherwise).

**How to work with him:**
- The **AI/agent/RAG backend is a learning exercise** — guide, explain design decisions, and let him understand every piece; don't just dump giant finished files without explanation. He wants to *master* this, not just ship it.
- The **frontend is NOT a learning goal** — he dislikes frontend work. Generate it fully and keep it minimal (a chat/progress view and report view is enough).
- He prefers direct, honest feedback over politeness. Tell him when an idea is bad and why.
- Push back if he drifts into "learn more tutorials first" mode — the agreed philosophy is: learn inside the project, milestone by milestone.
- LLM provider used so far: **Groq (Llama 3.3 70B)** for chat + **Google Gemini embeddings** (free tiers). Stay on free/cheap tiers unless he says otherwise.

---

## 2. Product Overview

### One-liner
An agentic AI advisor that tells Pakistani students **which universities they can actually get into, what it will cost, and which scholarships they qualify for** — with citations from official documents and deterministic merit-formula math.

### Problem
University admission info in Pakistan is scattered across PDFs, outdated websites, and Facebook groups. Merit formulas differ per university (matric %, FSc %, entry-test weightings). Deadlines change every cycle. Generic chatbots (ChatGPT/Gemini/Claude) **hallucinate** this data — they don't have the current prospectuses, can't compute merit aggregates reliably, and forget the student between sessions.

### Why this beats "just ask ChatGPT" (the moats — protect these in every design decision)
1. **Curated private corpus:** ingested, versioned prospectuses/fee schedules/scholarship docs that aren't on the open web in usable form. Every answer is cited against them.
2. **Deterministic domain logic:** merit aggregate calculators per university implemented as **code tools**, not LLM guesses.
3. **Persistence & workflow:** long-term student profile (marks, budget, city, preferences) via LangGraph Store; tracked deadlines; works like an advisor across sessions, not a one-off chat.

### Target users
Pakistani FSc/A-level students applying to universities (initially CS/engineering programs at ~5 seed universities: FAST, NUST, COMSATS, GIKI, Air University — adjustable).

### CV framing
Position as a **"domain-grounded advisory engine, launched for university admissions"** — the architecture is domain-agnostic; admissions is the launch market.

### Non-goals (v1)
- No application form submission / no acting on the student's behalf.
- No mobile app. Minimal web UI only.
- No multimodal RAG, no fine-tuning.
- Not covering every university in Pakistan — 5 done excellently beats 50 done badly.

---

## 3. Core Features (v1)

1. **Profile-aware Q&A with citations** — "Can I get into FAST CS with my marks?" → answer grounded in corpus + eligibility tool output, with document/page citations.
2. **Merit aggregate calculation** — per-university formulas as deterministic Python tools; agent selects and runs the right one from the stored profile.
3. **University shortlist & comparison** — given profile + budget + city constraints, produce a ranked, cited comparison (fees, merit last year vs. student's aggregate, scholarships).
4. **Scholarship matching** — eligibility-filtered scholarship list from the corpus.
5. **Live deadline lookup** — web-search worker (Tavily) for current-cycle dates when the corpus may be stale; answers must label corpus vs. web sources.
6. **Persistent student profile** — collected conversationally, stored in LangGraph Store, editable, remembered across sessions.
7. **Answer quality gate** — critic node grades every substantive answer against retrieved sources before it reaches the user; failed answers loop back (max 2 retries).
8. **Human-in-the-loop checkpoint** — before saving/overwriting profile facts, `interrupt` for user confirmation.

---

## 4. Architecture

```
                        ┌──────────────────────┐
  Student ── FastAPI ──▶│  SUPERVISOR (LangGraph)│◀── Store: student profile
  (SSE streaming)       └──┬──────┬──────┬──────┘    (Postgres, cross-thread)
                           │      │      │
              ┌────────────▼┐  ┌──▼────────┐  ┌─▼──────────────┐
              │ RAG worker  │  │ Web worker │  │ Eligibility     │
              │ (subgraph)  │  │ (Tavily)   │  │ subgraph        │
              │ hybrid      │  │ live       │  │ merit-formula   │
              │ retrieval + │  │ deadlines  │  │ tools (pure py) │
              │ reranking   │  │            │  │                 │
              └──────┬──────┘  └──────┬─────┘  └─┬───────────────┘
                     └────────┬───────┴──────────┘
                       ┌──────▼──────┐   reject → retry loop (max 2)
                       │   CRITIC    │──────────────────────────┐
                       └──────┬──────┘                          │
                           approved                             │
                       ┌──────▼──────────────┐                  │
                       │ Cited answer → user │◀─────────────────┘
                       └─────────────────────┘
```

- **Supervisor:** LangGraph graph; routes via `Command(goto=..., update=...)` handoffs. Each worker is a compiled subgraph.
- **RAG worker (agentic RAG):** query rewrite → hybrid retrieval (BM25 + pgvector) → cross-encoder rerank → relevance grading → answer or fallback-to-web (corrective RAG loop).
- **Ingestion pipeline (offline, separate from serving):** PDF → parse → contextual chunking (chunk + LLM-generated situational context, as in his `contextual_retrieval.py`) → embed → pgvector. Tracks corpus versions/sources in a metadata table.
- **Persistence (one Postgres instance for everything):** `PostgresSaver` (per-thread checkpoints), `PostgresStore` (student profiles, cross-thread), pgvector (chunks).
- **API:** FastAPI; `POST /chat` streams tokens + agent progress events over SSE (`stream_mode="messages"` / `astream_events`); profile CRUD endpoints; ingestion trigger endpoint (admin).
- **Observability:** LangSmith tracing on everything from day one.
- **Evals (first-class, in `evals/`):** golden Q&A dataset (~30–50 questions with known answers from the corpus, incl. adversarial/out-of-corpus ones), RAGAS or LangSmith evaluators for faithfulness, answer relevance, retrieval quality; eligibility tools get plain pytest unit tests. Scores published in README.
- **Frontend:** minimal React/Next.js chat UI showing streamed agent progress ("searching prospectuses… running FAST merit formula…"), final cited answer, and profile panel. Generated, not hand-crafted.

### Stack
| Layer | Choice |
|---|---|
| Agents | LangGraph (supervisor + subgraphs) |
| LLM | Groq Llama 3.3 70B (free tier) via `init_chat_model` |
| Embeddings | Google Gemini embeddings |
| Reranker | bge-reranker (local, free) or Cohere Rerank trial |
| DB | Postgres + pgvector (docker-compose), `PostgresSaver`, `PostgresStore` |
| API | FastAPI + SSE |
| Search | Tavily |
| Observability | LangSmith |
| Evals | RAGAS / LangSmith evals + pytest |
| Frontend | Next.js (minimal, generated) |

### Repo layout
```
rehnuma/
├── docker-compose.yml        # postgres + pgvector
├── pyproject.toml
├── PRD.md                    # this file
├── ingestion/                # PDF → chunks → pgvector (offline)
├── agents/                   # supervisor, workers/, tools/ (merit formulas), critic
├── api/                      # FastAPI app, SSE streaming, profile routes
├── evals/                    # golden dataset, RAGAS/LangSmith evals
├── corpus/                   # raw PDFs (gitignored if large) + sources.md
└── frontend/                 # minimal Next.js UI
```

---

## 5. Milestones (~4 weeks; finish each before starting the next)

**M1 — Skeleton on real infrastructure (week 1).** docker-compose Postgres+pgvector up; ingestion pipeline ingests 4–5 universities' PDFs into pgvector; one simple graph on `PostgresSaver` answers over the corpus through a FastAPI endpoint; LangSmith tracing on. *Done when: kill the server, restart, resume the same thread_id, and get a cited answer.*

**M2 — Agent system (week 2).** Supervisor + RAG worker (hybrid + rerank + grading + corrective loop) + web worker + eligibility subgraph with 5 merit-formula tools (pytest-covered) + critic loop + Store-backed student profile with HITL confirmation. *Done when: "Can I get into FAST CS? My marks are…" runs the full pipeline end-to-end.*

**M3 — Streaming, evals, UI (week 3).** SSE token + progress streaming; golden eval dataset + RAGAS/LangSmith eval run with scores; minimal frontend wired up. *Done when: eval report exists and the UI shows live agent progress.*

**M4 — Hardening & polish (week 4).** Error handling/retries/rate-limit fallbacks; README with architecture diagram, eval scores, demo GIF; deploy (Railway/VPS); 3–5 real students test it. *Done when: a stranger can use the deployed app and the README sells the project.*

---

## 6. Key risks
- **Corpus staleness:** admission data changes every cycle → every chunk carries source + date metadata; answers state data vintage; web worker covers current-cycle facts.
- **Hallucinated numbers:** merit/fees must come from tools or cited chunks, never from the LLM's head — the critic explicitly checks this.
- **Scope creep:** 5 universities, CS/engineering programs, v1 features only. New ideas go to a `BACKLOG.md`, not the sprint.
- **Free-tier rate limits (Groq):** design for retries/backoff early; keep an eye on token usage per request.
