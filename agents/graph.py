"""
M1 skeleton graph: one node, retrieval + cited answer, no supervisor/workers yet
(that's M2). Goal for now: prove the corpus can actually answer a real question
with citations, end to end.
"""

from dotenv import load_dotenv

load_dotenv()

import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.postgres import PostgresSaver

from ingestion.store import get_vector_store

llm = init_chat_model("groq:llama-3.3-70b-versatile")
store = get_vector_store()

# raw psycopg (used by PostgresSaver) wants a plain postgresql:// string,
# not the "+psycopg" SQLAlchemy dialect suffix langchain_postgres needs.
CHECKPOINT_DB_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")


def get_retriever(k: int = 3):
    
    return store.as_retriever(search_kwargs={"k": k})


retriever = get_retriever()

def make_context_string(docs: list) -> str:
    """
    Build a context string from the retrieved chunks — include enough
    to cite (university, page) alongside each chunk's text.
    """
    context = ""
    for doc in docs:
        university = doc.metadata.get("university", "Unknown University")
        page = doc.metadata.get("page", "Unknown Page")
        context += f"[{university}, Page {page}]: {doc.page_content}\n\n"
    return context.strip()

PROMPT_TEMPLATE = """You are answering questions about Pakistani university admissions using ONLY the context below. Footnotes and asterisked notes in the context are valid information — treat them the same as regular text.

Context:
{context}

Question: {question}

Answer using the context above, citing university and page for each claim. Only say "I don't know" if the context truly contains no relevant information."""


def answer(state: MessagesState) -> dict:
    message = state["messages"][-1].content
    docs = retriever.invoke(message)
    context = make_context_string(docs)
    response = llm.invoke(PROMPT_TEMPLATE.format(context=context, question=message))
    return {"messages": [response]}


def build_graph(checkpointer):
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("answer", answer)
    graph_builder.set_entry_point("answer")
    graph_builder.set_finish_point("answer")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    with PostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer:
        checkpointer.setup()  # idempotent — safe to call on every run
        graph = build_graph(checkpointer)

        config = {"configurable": {"thread_id": "test-thread-1"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "What is FAST's merit formula for BS Engineering?"}]},
            config=config,
        )
        print(result["messages"][-1].content)
