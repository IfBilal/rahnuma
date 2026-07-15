"""
Supervisor: routes a question to either the eligibility subgraph (merit
calculation, given marks) or the RAG worker (general corpus questions),
using Command(goto=...) handoffs per the PRD architecture.

Web worker and critic aren't built yet — routing is binary for now.
"""

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.types import Command

from agents.eligibility import eligibility_agent
from agents.rag_worker import build_rag_graph

llm = init_chat_model("groq:llama-3.3-70b-versatile")

# raw psycopg (used by PostgresSaver) wants a plain postgresql:// string,
# not the "+psycopg" SQLAlchemy dialect suffix langchain_postgres needs.
CHECKPOINT_DB_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")

ROUTING_PROMPT = """Classify the user's question into exactly one category:

"eligibility" — the user provides marks/percentages and wants a merit aggregate
calculated for a specific university/program.

"general" — anything else: fees, scholarships, deadlines, eligibility criteria
explanations, comparisons, or any question not providing marks to calculate with.

Question: {question}

Answer with exactly one word: "eligibility" or "general"."""


def router(state: MessagesState) -> Command[Literal["eligibility_worker", "rag_worker"]]:
    question = state["messages"][-1].content
    response = llm.invoke(ROUTING_PROMPT.format(question=question))
    decision = response.content.strip().lower()
    goto = "eligibility_worker" if "eligibility" in decision else "rag_worker"
    return Command(goto=goto)


def eligibility_worker(state: MessagesState) -> dict:
    result = eligibility_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


rag_graph = build_rag_graph()


def rag_worker(state: MessagesState) -> dict:
    result = rag_graph.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def build_supervisor_graph(checkpointer=None):
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("router", router)
    graph_builder.add_node("eligibility_worker", eligibility_worker)
    graph_builder.add_node("rag_worker", rag_worker)

    graph_builder.add_edge(START, "router")
    graph_builder.set_finish_point("eligibility_worker")
    graph_builder.set_finish_point("rag_worker")

    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_supervisor_graph()

    print("--- Eligibility question ---")
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "My matric is 90%, FSc is 85%, test 80%. What's my FAST BS Engineering merit?"}]}
    )
    print(result["messages"][-1].content)

    print("\n--- General question ---")
    result2 = graph.invoke(
        {"messages": [{"role": "user", "content": "What scholarships does GIKI offer?"}]}
    )
    print(result2["messages"][-1].content)
