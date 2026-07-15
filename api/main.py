"""
M2 FastAPI wrapper around the full supervisor system. Two endpoints:

POST /chat         — ask a question, get a cited/computed answer. If the
                      message also mentioned profile facts (marks, budget,
                      city), the response additionally flags that a profile
                      update is waiting for confirmation.
POST /chat/confirm — approve or decline that pending profile update.

Both the PostgresSaver (per-thread conversation state) and PostgresStore
(cross-thread student profile) connections are opened once at server
startup via `lifespan` and stay open for the server's whole lifetime.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.types import Command
from pydantic import BaseModel

from agents.supervisor import CHECKPOINT_DB_URL, build_supervisor_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer, \
         PostgresStore.from_conn_string(CHECKPOINT_DB_URL) as store:
        checkpointer.setup()  # idempotent — safe to call on every startup
        store.setup()
        app.state.graph = build_supervisor_graph(checkpointer, store)
        yield
    # the `with` block closes both Postgres connections here, on shutdown


app = FastAPI(lifespan=lifespan)


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


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.thread_id}}
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
    config = {"configurable": {"thread_id": request.thread_id}}
    result = app.state.graph.invoke(Command(resume=request.approved), config=config)
    return ConfirmResponse(profile_result=result["profile_result"])
