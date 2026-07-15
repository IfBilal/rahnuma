"""
Student profile: long-term, cross-thread memory via PostgresStore (unlike
PostgresSaver, which is per-conversation, Store persists per-STUDENT across
every conversation they ever have). Before saving or overwriting any profile
fact, the graph interrupts and waits for explicit human confirmation — the
PRD's HITL requirement.
"""

from dotenv import load_dotenv

load_dotenv()

import os
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.postgres import PostgresStore
from langgraph.types import interrupt, Command
from typing_extensions import Annotated

llm = init_chat_model("groq:llama-3.3-70b-versatile")

CHECKPOINT_DB_URL = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")

EXTRACT_PROMPT = """Extract any student profile facts mentioned in this message:
marks (matric_pct, fsc_pct), budget (in PKR), city, or preferred universities.

Message: {message}

Respond with a JSON object using only the keys that are actually mentioned,
e.g. {{"matric_pct": 90, "city": "Lahore"}}. If nothing is mentioned, respond
with an empty JSON object: {{}}"""


class ProfileState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    extracted_facts: dict


def extract_facts(state: ProfileState) -> dict:
    message = state["messages"][-1].content
    response = llm.invoke(EXTRACT_PROMPT.format(message=message))

    import json

    try:
        facts = json.loads(response.content.strip())
    except json.JSONDecodeError:
        facts = {}

    return {"extracted_facts": facts}


def confirm_and_save(state: ProfileState, store) -> dict:
    facts = state["extracted_facts"]
    if not facts:
        return {"messages": [AIMessage(content="Nothing profile-worthy to save from that message.")]}

    existing = store.get(("students", state["user_id"]), "profile")
    existing_data = existing.value if existing else {}

    approved = interrupt(
        {
            "action": "confirm_profile_update",
            "current_profile": existing_data,
            "proposed_changes": facts,
        }
    )

    if not approved:
        return {"messages": [AIMessage(content="Okay, I won't save those details to your profile.")]}

    updated = {**existing_data, **facts}
    store.put(("students", state["user_id"]), "profile", updated)
    return {"messages": [AIMessage(content=f"Saved to your profile: {facts}")]}


def build_profile_graph(checkpointer=None, store=None):
    graph_builder = StateGraph(ProfileState)
    graph_builder.add_node("extract_facts", extract_facts)
    graph_builder.add_node("confirm_and_save", confirm_and_save)
    graph_builder.add_edge(START, "extract_facts")
    graph_builder.add_edge("extract_facts", "confirm_and_save")
    return graph_builder.compile(checkpointer=checkpointer, store=store)


if __name__ == "__main__":
    with PostgresStore.from_conn_string(CHECKPOINT_DB_URL) as store:
        store.setup()

        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()  # just for this standalone test
        graph = build_profile_graph(checkpointer=checkpointer, store=store)

        config = {"configurable": {"thread_id": "profile-test-1"}}
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "My matric was 88% and my FSc was 91%, budget is around 500000 PKR"}],
                "user_id": "student-42",
            },
            config=config,
        )

        # graph paused at interrupt() — inspect what it's asking us to confirm
        print("--- Interrupted, waiting for confirmation ---")
        print(result["__interrupt__"][0].value)

        print("\n--- Resuming with approval ---")
        final = graph.invoke(Command(resume=True), config=config)
        print(final["messages"][-1].content)

        print("\n--- Verifying it actually persisted ---")
        saved = store.get(("students", "student-42"), "profile")
        print(saved.value)
