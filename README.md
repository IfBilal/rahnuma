# Rahnuma — University Admissions & Scholarship Advisor

Rahnuma is a domain-grounded advisory engine for Pakistani university admissions. It answers against a curated corpus of official prospectuses, calculates merit aggregates with deterministic Python, remembers confirmed student facts across conversations, and marks live-web answers as web-sourced.

The point is not to make a generic admissions chatbot. The point is to make every high-stakes number traceable to either **code** (merit arithmetic) or a **source** (prospectus/web result).

## What a student can do

- Ask cited questions about admission requirements, programs, fees, scholarships, and policies.
- Provide marks and receive a deterministic aggregate for FAST, NUST, COMSATS, GIKI, or UET.
- Ask follow-ups; the RAG worker rewrites them into standalone retrieval queries.
- Create an account, then let Rahnuma remember marks, budget, city, and preferred universities—but only after an explicit confirmation.
- See live progress and streamed answer tokens in the Next.js UI.

## System map

```mermaid
flowchart LR
    UI["Next.js UI\nchat · profile · confirmation"]
    API["FastAPI\n/chat · /chat/stream · /chat/confirm"]
    PG[("Postgres + pgvector\nchunks · checkpoints · profiles")]
    LS["LangSmith\noptional traces"]

    UI -->|JSON or SSE| API
    API --> SUP
    API <--> PG
    SUP["LangGraph supervisor"] <--> PG
    SUP --> RAG["RAG worker subgraph"]
    SUP --> ELIG["Eligibility agent"]
    SUP --> PROFILE["Profile capture branch"]
    RAG --> PG
    RAG --> WEB["Tavily web worker subgraph"]
    ELIG --> TOOLS["Pure-Python merit tools"]
    SUP -. optional tracing .-> LS
```

## The complete LangGraph execution graph

This is the graph compiled by `build_supervisor_graph()` in [`agents/supervisor.py`](agents/supervisor.py). The answer branch and profile branch deliberately begin in parallel from `START`.

```mermaid
flowchart TD
    START((START))
    END((END))

    START --> ROUTER["router\nLLM intent classifier"]
    ROUTER -->|marks + target university + aggregate request| EW["eligibility_worker\ninvokes eligibility agent"]
    ROUTER -->|everything else| RW["rag_worker\ninvokes RAG subgraph"]

    EW --> CRITIC["critic\nsource-faithfulness gate"]
    RW --> CRITIC
    CRITIC -->|approved| END
    CRITIC -->|rejected, retry < 2| EW
    CRITIC -->|rejected, retry < 2| RW
    CRITIC -->|rejected, retry limit reached| END

    START --> EXTRACT["extract_facts\nLLM JSON extraction"]
    EXTRACT --> SAVE["confirm_and_save\ninterrupt before mutation"]
    SAVE -. profile confirmation is independent of answer delivery .-> END
```

### Supervisor state

| Field | Role | Writer(s) |
|---|---|---|
| `messages` | Conversation and final worker answer. Uses LangGraph's `add_messages` reducer. | User input, eligibility worker, RAG worker, critic failure path |
| `context` | Source material used by the latest answer. | Eligibility worker or RAG worker |
| `current_worker` | Remembers the worker to retry if the critic rejects the answer. | Eligibility worker or RAG worker |
| `retry_count` | Number of critic-directed retries. | Critic |
| `user_id` | Profile namespace identifier. | API input |
| `extracted_facts` | Candidate profile update from the profile branch. | `extract_facts` |
| `profile_result` | Save/decline status; kept separate from messages to avoid parallel-write ordering ambiguity. | `confirm_and_save` |

### Every supervisor node

#### 1. `router`

- **Input:** latest user message.
- **Model:** Groq `llama-3.3-70b-versatile`.
- **Job:** returns exactly `eligibility` or `general`.
- **Decision:** it sends a question to `eligibility_worker` only if the user supplied marks/percentages and asked for a specific university/program aggregate. Fees, scholarships, deadlines, explanations, comparisons, and incomplete-mark questions go to `rag_worker`.
- **Why an LLM here:** intent phrasing is varied; the output space is tiny and constrained.

#### 2. `eligibility_worker`

- **Input:** full message history.
- **Job:** invokes the nested eligibility agent with the parent runnable configuration, so LangSmith and SSE can observe the nested tool calls.
- **Output:** final agent response, a compact `context` string containing the selected tool call and its numerical result, and `current_worker="eligibility_worker"`.
- **Why it records tool output:** the critic can check that the displayed aggregate was actually returned by deterministic code.

#### 3. `rag_worker`

- **Input:** full message history.
- **Job:** invokes the compiled RAG subgraph below.
- **Output:** the RAG answer, source context used for grounding, and `current_worker="rag_worker"`.

#### 4. `critic`

- **Input:** latest human question, latest proposed answer, and the worker’s `context`.
- **Model:** Groq `llama-3.3-70b-versatile`.
- **Job:** returns `approved` only when the answer is faithful to its supplied evidence: no invented facts, numbers, or unsupported claims.
- **Approved:** ends the answer branch.
- **Rejected before retry limit:** sends execution straight back to the same worker; it does *not* rerun routing, which avoids accidental worker switching and wasted calls.
- **Rejected after two retries:** ends with an explicit “could not be fully verified” disclaimer. It never silently presents that answer as verified.

#### 5. `extract_facts`

- **Runs in parallel with routing and answering.** It does not delay the selection of an answer worker.
- **Input:** latest message.
- **Extracts only:** `matric_pct`, `fsc_pct`, budget in PKR, city, and preferred universities.
- **Output:** an empty object if there are no facts; otherwise candidate facts parsed from an LLM JSON response.
- **Failure behavior:** malformed JSON becomes `{}` rather than breaking the chat request.

#### 6. `confirm_and_save`

- **Input:** `extracted_facts`, `user_id`, and injected `PostgresStore`.
- **No facts:** returns immediately with an empty `profile_result`.
- **Facts found:** reads the existing profile from `("students", user_id) / "profile"`, then calls LangGraph `interrupt()` with the current profile and proposed changes.
- **On `POST /chat/confirm`:** `Command(resume=True)` merges new facts over existing facts and writes them to the Store. `False` leaves storage untouched.
- **Important:** the graph checkpoint preserves the interrupt state, so confirmation happens against the exact pending update—not a reconstructed guess.

## Eligibility agent subgraph

The eligibility worker invokes a LangChain ReAct agent defined in [`agents/eligibility.py`](agents/eligibility.py). The LLM selects a tool and extracts input percentages; it does **not** perform arithmetic itself.

```mermaid
flowchart LR
    Q["User marks + target university"] --> A["Eligibility ReAct agent"]
    A -->|selects one tool| T{"Deterministic merit tool"}
    T --> FAST["FAST\nBS or Engineering"]
    T --> COM["COMSATS"]
    T --> GIKI["GIKI"]
    T --> UET["UET"]
    T --> NUST["NUST"]
    FAST --> R["Tool result → agent answer"]
    COM --> R
    GIKI --> R
    UET --> R
    NUST --> R
```

| Tool | Inputs | Formula source in code |
|---|---|---|
| `calculate_fast_merit` | matric %, FSc %, admission-test %, `bs`/`engineering` | `fast_merit()` |
| `calculate_comsats_merit` | matric %, FSc %, NTS % | `comsats_merit()` |
| `calculate_giki_merit` | SSC/O-level %, GIKI-test % | `giki_merit()` |
| `calculate_uet_merit` | matric %, FSc %, ECAT % | `uet_merit()` |
| `calculate_nust_merit` | matric %, HSSC/FSc %, NET % | `nust_merit()` |

The tools live in [`agents/tools/merit.py`](agents/tools/merit.py). They are pure functions with pytest coverage. Formula weights are versioned code data and must be checked against a new admission cycle before being changed.

## RAG worker subgraph

The RAG worker is compiled by `build_rag_graph()` in [`agents/rag_worker.py`](agents/rag_worker.py). It is the source-backed path for non-aggregate questions.

```mermaid
flowchart TD
    RS((START)) --> REWRITE["rewrite_query\nmake follow-up standalone"]
    REWRITE --> RETRIEVE["retrieve\nhybrid candidates + rerank"]
    RETRIEVE --> GRADE["grade\ncorpus relevance check"]
    GRADE -->|relevant| ANSWER["answer\ncited corpus-only response"]
    GRADE -->|not relevant| FALLBACK["fallback\nweb worker subgraph"]
    ANSWER --> RE((END))
    FALLBACK --> RE
```

### Every RAG node

#### 1. `rewrite_query`

- **One-message conversation:** uses the user message untouched.
- **Follow-up conversation:** asks the model to transform the final turn into a standalone search query using the earlier conversation as context.
- **Example:** “How does that compare to their BBA formula?” becomes a question that explicitly names the university and preceding formula.

#### 2. `retrieve`

This node performs three stages:

1. **Vector retrieval:** pgvector runs semantic similarity search over the `rahnuma_corpus` collection with local `BAAI/bge-small-en-v1.5` embeddings.
2. **Lexical retrieval:** BM25 searches the same corpus text. It is built from the already-ingested `langchain_pg_embedding` rows—not by reopening PDFs while users wait.
3. **Ensemble + rerank:** `EnsembleRetriever` combines vector and BM25 candidates with equal `0.5 / 0.5` weights. `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder then scores the query together with each candidate and retains the top 10.

The final `context` carries text prefixed with citation metadata in this form:

```text
[university, source file, Page N]: chunk text
```

That exact context is supplied both to the answer node and later to the supervisor critic.

#### 3. `grade`

- **Input:** standalone query plus reranked context.
- **Output:** one boolean, `is_relevant`.
- **Decision:** a `yes` routes to corpus answering; a `no` routes to live web fallback. This is the corrective-RAG guard against confidently answering from irrelevant chunks.

#### 4. `answer`

- **Constraint:** prompts the model to use *only* retrieved context.
- **Output requirement:** cite university and page for every claim.
- **Streaming:** this is one of the user-visible answer nodes; its tokens are emitted over SSE.

#### 5. `fallback`

- **When:** the corpus relevance grader says no.
- **Action:** invokes the web worker with the rewritten query.
- **Grounding:** replaces the RAG context with formatted web results so the supervisor critic checks the final web-backed answer against the correct evidence.

### Lazy runtime initialization

The pgvector store, BM25 index, web graph, and ONNX cross-encoder are created by `get_retrieval_runtime()` only when the first RAG request arrives. This keeps `/health`, profile endpoints, and API import usable while Postgres is unavailable, and avoids loading large models for merit-only requests.

## Web worker subgraph

The web worker in [`agents/web_worker.py`](agents/web_worker.py) is intentionally narrow: it is a live-information fallback, not the default source of truth.

```mermaid
flowchart LR
    WSTART((START)) --> SEARCH["search\nTavilySearch, max 5 results"]
    SEARCH --> WANSWER["answer\nweb-only, labelled response"]
    WANSWER --> WEND((END))
```

| Node | Detail |
|---|---|
| `search` | Sends the user/re-written query to Tavily and stores URL + content for up to five results. |
| `answer` | Uses only those results, labels the response as web-sourced, and tells the student to verify time-sensitive facts on the official admissions site. |

## Data, ingestion, and persistence

### Offline ingestion path

```mermaid
flowchart LR
    PDF["Official PDF"] --> LOAD["loader.py\npypdf, page-level text"]
    LOAD --> CHUNK["chunker.py\n1,000 chars · 150 overlap"]
    CHUNK --> META["Metadata\nuniversity · filename · page"]
    META --> EMBED["LocalEmbeddings\nFastEmbed / BGE small"]
    EMBED --> VECTOR[("PGVector\nrahnuma_corpus")]
```

- Raw documents live under `corpus/<university>/`.
- [`corpus/sources.md`](corpus/sources.md) records source URL, fetch date, admission cycle, and OCR notes.
- Every chunk preserves university, source filename, and original page number for citation.
- `ingestion/run.py` walks each university directory and writes chunks to pgvector.
- The current local collection contains 2,265 chunks.

### One Postgres instance, three responsibilities

| Component | What it persists | Scope |
|---|---|---|
| `PGVector` / `rahnuma_corpus` | Embedded prospectus chunks and metadata | Shared corpus |
| `PostgresSaver` / `AsyncPostgresSaver` | LangGraph state, messages, retries, pending interrupts | Per `thread_id` conversation |
| `PostgresStore` / `AsyncPostgresStore` | Confirmed profile under `("students", user_id) / "profile"` | Cross-thread, per student |

The FastAPI lifespan creates sync saver/store for normal JSON endpoints and async saver/store for SSE. They point to the same Postgres database, so streaming and non-streaming requests share the same truth.

## Authentication and profile ownership

Clerk owns registration, sign-in, password handling, sessions, and the browser
session token. Rahnuma never stores a password or mints its own user JWT.

```text
Clerk sign-up/sign-in UI
        ↓ Clerk session token (Bearer header)
Next.js frontend ──────────────────────────> FastAPI
                                               ↓ verifies token/JWKS
                                         Clerk user ID (`sub`)
                                               ↓
                         PostgresStore: ("students", clerk_user_id) / "profile"
```

Every profile and chat endpoint is protected. The API verifies the Clerk token,
reads the stable Clerk `sub` claim, and passes that ID into the LangGraph
profile branch. A browser can no longer choose another user's `user_id` in a
request body. `thread_id` remains a client-generated conversation identifier;
the profile owner always comes from the verified Clerk session.

For local development, use a Clerk development instance (`pk_test_...` and
`sk_test_...`). This is the same architecture used after deployment; only the
environment keys and allowed origin change.

## API and streaming contract

| Endpoint | Purpose |
|---|---|
| `GET /health` | Lightweight liveness response: `{ "status": "ok" }`. |
| `GET /profiles/me` | Reads the signed-in student’s confirmed profile. |
| `POST /chat` | Requires Clerk auth; runs the synchronous graph and returns an answer plus any pending profile confirmation. |
| `POST /chat/confirm` | Requires Clerk auth; resumes the signed-in user's interrupted profile branch with `approved: true` or `false`. |
| `POST /chat/stream` | Requires Clerk auth; runs the async graph and emits Server-Sent Events. |

`/chat/stream` emits JSON SSE frames with three event shapes:

```text
{ "type": "progress", "message": "Searching university prospectuses..." }
{ "type": "token", "content": "..." }
{ "type": "done", "answer": "...", "profile_confirmation_pending": true }
```

Internal routing, grading, and critic model tokens are not streamed to the user. Only real answer-node tokens are exposed.

## Frontend

The generated Next.js app lives in [`frontend/`](frontend/). It provides:

- A responsive admissions chat interface.
- Live agent-progress messages and token streaming.
- Clerk sign-up/sign-in and account menu.
- A persistent browser-side `thread_id` UUID for conversation checkpoints.
- A profile panel backed by authenticated `GET /profiles/me`.
- A proposed-profile confirmation card calling `POST /chat/confirm`.
- Starter prompts, error states, and an explicit source-trust explanation.

The frontend uses `NEXT_PUBLIC_API_BASE_URL`, defaulting to `http://localhost:8000`. See [`frontend/.env.example`](frontend/.env.example).

## LangSmith observability

LangGraph and LangChain trace automatically when these optional values are present before FastAPI starts:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rahnuma-local
```

Each API graph invocation uses a meaningful run name (`rahnuma-chat`, `rahnuma-chat-stream`, or `rahnuma-profile-confirmation`), `rahnuma` + route tags, and non-sensitive metadata (`thread_id`, `api_route`). Profile contents are intentionally not added to trace metadata.

Traces expose the nested supervisor, worker, tool, and model-call tree—useful for debugging routing, source quality, tool choice, and critic retries. Treat captured prompts and outputs as student data. Create an API key in [LangSmith settings](https://smith.langchain.com/settings).

## Local setup

### 1. Configure environment

Copy [`.env.example`](.env.example) to `.env`, then provide the required values:

```text
DATABASE_URL=postgresql+psycopg://rahnuma:rahnuma_dev@localhost:5432/rahnuma
GROQ_API_KEY=...
TAVILY_API_KEY=...
CLERK_SECRET_KEY=sk_test_...
CLERK_PUBLISHABLE_KEY=pk_test_...
LANGSMITH_TRACING=true             # optional
LANGSMITH_API_KEY=...              # optional
LANGSMITH_PROJECT=rahnuma-local    # optional
```

`GOOGLE_API_KEY` and `HF_TOKEN` are used only by their respective configured model/download workflows.

In `frontend/.env.local`, set:

```text
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
```

### 2. Start the database

```bash
docker compose up -d
```

### 3. Ingest the corpus if needed

```bash
uv run python -m ingestion.run
```

Do this only for an empty/new collection; repeated ingestion currently adds duplicate chunks.

### 4. Run backend and frontend

```bash
# terminal 1
uv run uvicorn api.main:app --reload

# terminal 2
npm --prefix frontend install
npm --prefix frontend run dev
```

Open `http://localhost:3000`. Swagger docs are available at `http://localhost:8000/docs`.

## Verification performed

```bash
uv run pytest -q
npm --prefix frontend run build
```

Current checks:

- 13 pytest tests pass: deterministic merit formulas, chunk metadata/citation behavior, API basics, and LangSmith graph configuration.
- Next.js production build passes.
- Postgres + pgvector connection verified with 2,265 indexed chunks.
- Real `/chat` eligibility flow verified: tool calculation → critic → profile interrupt.
- Real profile confirmation and persistence verified.
- Real corpus-backed RAG response verified with a FAST prospectus citation.
- Real `/chat/stream` request verified: progress, token, and done SSE events.
- LangSmith project connectivity verified with `rahnuma-local`.

## Known boundaries and next milestone

- Merit weights are cycle-dependent. Refresh `agents/tools/merit.py` only after verifying the new official prospectus.
- Corpus facts have a vintage; deadline-like information can require web fallback and official-site verification.
- The system is intentionally scoped to five seed universities and CS/engineering-oriented admissions.
- LangSmith tracing is configured; the next observability milestone is a golden question dataset and evaluation runs for retrieval quality, answer relevance, and faithfulness.
- Deployment is intentionally out of scope for the current local milestone.
