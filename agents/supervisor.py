"""
Supervisor: routes a question to either the eligibility subgraph (merit
calculation, given marks) or the RAG worker (general corpus questions),
using Command(goto=...) handoffs per the PRD architecture. Every answer
passes through a critic before reaching the user — the PRD's "answer
quality gate": grade against sources, retry up to 2x on rejection.
"""

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import Annotated

from agents.eligibility import eligibility_agent
from agents.rag_worker import build_rag_graph
from agents.profile import extract_facts, confirm_and_save

llm = init_chat_model("groq:llama-3.3-70b-versatile")

# raw psycopg (used by PostgresSaver) wants a plain postgresql:// string,
# not the "+psycopg" SQLAlchemy dialect suffix langchain_postgres needs.
CHECKPOINT_DB_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")

MAX_RETRIES = 2


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str  # sources the last worker used, so the critic can check grounding
    retry_count: int
    current_worker: str  # which worker produced the last answer, so critic can retry it directly
    user_id: str  # identifies the student, for cross-thread profile memory
    extracted_facts: dict  # profile facts found in this message, if any
    profile_result: str  # status of the profile save, kept separate from
    # `messages` since two independent branches (answer + profile) both
    # writing to messages makes "last message" ambiguous — see profile.py


ROUTING_PROMPT = """Classify the user's question into exactly one category:

"eligibility" — the user provides marks/percentages and wants a merit aggregate
calculated for a specific university/program.

"general" — anything else: fees, scholarships, deadlines, eligibility criteria
explanations, comparisons, or any question not providing marks to calculate with.

Question: {question}

Answer with exactly one word: "eligibility" or "general"."""


def router(state: SupervisorState) -> Command[Literal["eligibility_worker", "rag_worker"]]:
    question = state["messages"][-1].content
    response = llm.invoke(ROUTING_PROMPT.format(question=question))
    decision = response.content.strip().lower()
    goto = "eligibility_worker" if "eligibility" in decision else "rag_worker"
    return Command(goto=goto)


def eligibility_worker(state: SupervisorState) -> dict:
    result = eligibility_agent.invoke({"messages": state["messages"]})

    # build a context string from the tool call + result, so the critic can
    # verify the final answer's number actually matches what the tool returned
    context_parts = []
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                context_parts.append(f"Tool called: {tc['name']}({tc['args']})")
        if m.type == "tool":
            context_parts.append(f"Tool result: {m.content}")

    return {
        "messages": [result["messages"][-1]],
        "context": "\n".join(context_parts),
        "current_worker": "eligibility_worker",
    }


rag_graph = build_rag_graph()


def rag_worker(state: SupervisorState) -> dict:
    result = rag_graph.invoke({"messages": state["messages"]})
    return {
        "messages": [result["messages"][-1]],
        "context": result.get("context", ""),
        "current_worker": "rag_worker",
    }


CRITIC_PROMPT = """Question: {question}

Sources used to answer:
{context}

Proposed answer:
{answer}

Is this answer faithful to the sources above — no fabricated facts, no
hallucinated numbers, no claims the sources don't support? Answer with
exactly one word: "approved" or "rejected"."""


def critic(state: SupervisorState) -> Command[Literal["eligibility_worker", "rag_worker", "__end__"]]:
    human_messages = [m for m in state["messages"] if m.type == "human"]
    question = human_messages[-1].content
    answer = state["messages"][-1].content
    retry_count = state.get("retry_count", 0)

    prompt = CRITIC_PROMPT.format(question=question, context=state.get("context", ""), answer=answer)
    response = llm.invoke(prompt)
    approved = "approved" in response.content.strip().lower()

    if approved:
        return Command(goto=END)

    if retry_count >= MAX_RETRIES:
        # gave up after exhausting retries — the answer is still unverified,
        # so say so explicitly instead of returning it as if it passed
        disclaimer = (
            "\n\nNote: this answer could not be fully verified against sources "
            "after multiple attempts — please double-check it independently."
        )
        return Command(goto=END, update={"messages": [AIMessage(content=answer + disclaimer)]})

    # rejected, retries remain — retry the SAME worker directly, no need to
    # re-classify the question through router again (same answer every time,
    # wastes an LLM call, and risks flipping to a different worker on retry)
    return Command(goto=state["current_worker"], update={"retry_count": retry_count + 1})


def build_supervisor_graph(checkpointer=None, store=None):
    graph_builder = StateGraph(SupervisorState)
    graph_builder.add_node("router", router)
    graph_builder.add_node("eligibility_worker", eligibility_worker)
    graph_builder.add_node("rag_worker", rag_worker)
    graph_builder.add_node("critic", critic)
    graph_builder.add_node("extract_facts", extract_facts)
    graph_builder.add_node("confirm_and_save", confirm_and_save)

    # answering the question and capturing profile facts run in parallel —
    # neither depends on the other, both read the same incoming message
    graph_builder.add_edge(START, "router")
    graph_builder.add_edge("eligibility_worker", "critic")
    graph_builder.add_edge("rag_worker", "critic")

    graph_builder.add_edge(START, "extract_facts")
    graph_builder.add_edge("extract_facts", "confirm_and_save")

    return graph_builder.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    with PostgresSaver.from_conn_string(CHECKPOINT_DB_URL) as checkpointer, \
         PostgresStore.from_conn_string(CHECKPOINT_DB_URL) as store:
        checkpointer.setup()
        store.setup()
        graph = build_supervisor_graph(checkpointer=checkpointer, store=store)

        config = {"configurable": {"thread_id": "supervisor-test-1"}}

        print("--- Eligibility question, with profile facts mentioned ---")
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "My matric is 90%, FSc is 85%, test 80%. What's my FAST BS Engineering merit?"}],
                "user_id": "student-99",
            },
            config=config,
        )
        print("Answer:", result["messages"][-1].content)
        print(f"(retries used: {result.get('retry_count', 0)})")
        if "__interrupt__" in result:
            print("Profile branch paused:", result["__interrupt__"][0].value)

            print("\n--- Resuming profile confirmation ---")
            resumed = graph.invoke(Command(resume=True), config=config)
            print("Profile branch:", resumed["profile_result"])
