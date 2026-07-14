"""
M1 FastAPI wrapper around the skeleton graph. One endpoint: POST /chat.

The PostgresSaver connection is opened once at server startup (via `lifespan`)
and stays open for the server's whole lifetime — not reopened per request.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from agents.graph import CHECKPOINT_DB_URL, build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    with PostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer:
        checkpointer.setup()  # idempotent — safe to call on every startup
        app.state.graph = build_graph(checkpointer)
        yield
    # the `with` block closes the Postgres connection here, on server shutdown


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.thread_id}}
    result = app.state.graph.invoke(
        {"messages": [{"role": "user", "content": request.question}]},
        config=config,
    )
    return ChatResponse(answer=result["messages"][-1].content)
