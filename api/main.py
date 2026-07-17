"""
M2/M3 FastAPI wrapper around the full supervisor system.

POST /chat         — ask a question, get a cited/computed answer. If the
                      message also mentioned profile facts (marks, budget,
                      city), the response additionally flags that a profile
                      update is waiting for confirmation.
POST /chat/confirm — approve or decline that pending profile update.
POST /chat/stream  — same as /chat, but streams progress + answer tokens
                      over SSE instead of returning one final JSON blob.

Two separate graph instances are built: one on sync PostgresSaver/Store
(for the simple non-streaming endpoints), one on the async variants (only
async checkpointers support astream_events, which the streaming endpoint
needs). Both point at the same Postgres tables, so conversation/profile
state is genuinely shared between them — they're just two connections to
the same underlying data, not two separate stores of truth.
"""

import json
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from pydantic import BaseModel

from agents.supervisor import CHECKPOINT_DB_URL, build_supervisor_graph

# node names whose *start* is worth telling the user about — translated to
# a friendly progress message. Anything not listed here (grade, critic,
# router's internals, etc.) stays silent — those are control-flow steps,
# not something a user needs a play-by-play of.
PROGRESS_MESSAGES = {
    "router": "Figuring out how to answer your question...",
    "rewrite_query": "Understanding your question...",
    "retrieve": "Searching university prospectuses...",
    "grade": "Checking if the corpus has a good answer...",
    "eligibility_worker": "Calculating your merit aggregate...",
    "tools": "Running the merit formula...",
    "fallback": "Corpus wasn't enough — searching the web...",
    "critic": "Double-checking the answer...",
}

# node names during which chat-model tokens are the REAL answer, worth
# streaming to the user. Every other LLM call (routing, grading, critic)
# is an internal single-word decision, not meant to be shown token-by-token.
ANSWER_NODES = {"answer", "model"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # sync and async context managers can't mix in one `async with` — nest them
    with PostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer, \
         PostgresStore.from_conn_string(CHECKPOINT_DB_URL) as store:
        checkpointer.setup()  # idempotent — safe to call on every startup
        store.setup()
        app.state.graph = build_supervisor_graph(checkpointer, store)
        app.state.store = store

        async with AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as async_checkpointer, \
                   AsyncPostgresStore.from_conn_string(CHECKPOINT_DB_URL) as async_store:
            await async_checkpointer.setup()
            await async_store.setup()
            app.state.async_graph = build_supervisor_graph(async_checkpointer, async_store)

            yield
    # both `with` blocks close all four Postgres connections here, on shutdown


app = FastAPI(lifespan=lifespan)

# The generated Next.js UI is served separately during local development.
# Keep the allowed origins explicit so browser requests work without opening
# this API to arbitrary websites.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    thread_id: str
    user_id: str


class ChatResponse(BaseModel):
    answer: str
    profile_confirmation_pending: bool = False
    profile_confirmation_details: dict | None = None


class ConfirmRequest(BaseModel):
    thread_id: str
    approved: bool


class ConfirmResponse(BaseModel):
    profile_result: str


class ProfileResponse(BaseModel):
    profile: dict


@app.get("/health")
def health() -> dict[str, str]:
    """A lightweight liveness endpoint for the UI and local diagnostics."""
    return {"status": "ok"}


def graph_config(thread_id: str, route: str) -> dict:
    """Attach useful, non-sensitive metadata to LangGraph/LangSmith runs.

    LangGraph automatically emits traces when LANGSMITH_TRACING=true and a
    LANGSMITH_API_KEY are present. We deliberately do not attach user profile
    contents here: prompts are already visible in a trace, and profile data
    should not become extra searchable metadata.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"rahnuma-{route}",
        "tags": ["rahnuma", route],
        "metadata": {"thread_id": thread_id, "api_route": route},
    }


@app.get("/profiles/{user_id}")
def get_profile(user_id: str) -> ProfileResponse:
    """Return the confirmed long-term profile for one student, if it exists."""
    item = app.state.store.get(("students", user_id), "profile")
    return ProfileResponse(profile=item.value if item else {})


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    config = graph_config(request.thread_id, "chat")
    result = app.state.graph.invoke(
        {"messages": [{"role": "user", "content": request.question}], "user_id": request.user_id},
        config=config,
    )

    if "__interrupt__" in result:
        return ChatResponse(
            answer=result["messages"][-1].content,
            profile_confirmation_pending=True,
            profile_confirmation_details=result["__interrupt__"][0].value,
        )

    return ChatResponse(answer=result["messages"][-1].content)


@app.post("/chat/confirm")
def confirm(request: ConfirmRequest) -> ConfirmResponse:
    config = graph_config(request.thread_id, "profile-confirmation")
    result = app.state.graph.invoke(Command(resume=request.approved), config=config)
    return ConfirmResponse(profile_result=result["profile_result"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        config = graph_config(request.thread_id, "chat-stream")
        in_answer_node = False

        async for event in app.state.async_graph.astream_events(
            {"messages": [{"role": "user", "content": request.question}], "user_id": request.user_id},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name")

            if kind == "on_chain_start" and name in ANSWER_NODES:
                in_answer_node = True
            elif kind == "on_chain_end" and name in ANSWER_NODES:
                in_answer_node = False

            if kind == "on_chain_start" and name in PROGRESS_MESSAGES:
                yield _sse({"type": "progress", "message": PROGRESS_MESSAGES[name]})

            elif kind == "on_chat_model_stream" and in_answer_node:
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield _sse({"type": "token", "content": chunk.content})

        # final state (after streaming completes) carries the interrupt flag
        # and full answer, same info /chat returns in one shot
        final_state = await app.state.async_graph.aget_state(config)
        result = final_state.values
        done_event = {"type": "done", "answer": result["messages"][-1].content}
        if final_state.tasks and any(t.interrupts for t in final_state.tasks):
            interrupt = next(t.interrupts[0] for t in final_state.tasks if t.interrupts)
            done_event["profile_confirmation_pending"] = True
            done_event["profile_confirmation_details"] = interrupt.value
        yield _sse(done_event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
