"""
Web worker: live web search (Tavily) for current-cycle facts the corpus may
not have yet (deadlines especially — prospectuses get published once per
cycle and go stale). Answers are explicitly labeled as web-sourced, per the
PRD requirement to distinguish corpus vs. web provenance to the user.
"""

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

llm = init_chat_model("groq:llama-3.3-70b-versatile")
search_tool = TavilySearch(max_results=5)


class WebState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: str


def search(state: WebState) -> dict:
    question = state["messages"][-1].content
    results = search_tool.invoke({"query": question})
    formatted = "\n\n".join(
        f"[{r['url']}]: {r['content']}" for r in results.get("results", [])
    )
    return {"search_results": formatted}


PROMPT_TEMPLATE = """Answer the question using ONLY the live web search results below.
Since these results are from the web (not the university's official prospectus corpus),
explicitly label your answer as web-sourced and remind the user to verify against the
official admissions website, since web info can be inaccurate.

Web search results:
{results}

Question: {question}"""


def answer(state: WebState) -> dict:
    question = state["messages"][-1].content
    prompt = PROMPT_TEMPLATE.format(results=state["search_results"], question=question)
    response = llm.invoke(prompt)
    return {"messages": [response]}


def build_web_graph(checkpointer=None):
    graph_builder = StateGraph(WebState)
    graph_builder.add_node("search", search)
    graph_builder.add_node("answer", answer)
    graph_builder.set_entry_point("search")
    graph_builder.add_edge("search", "answer")
    graph_builder.set_finish_point("answer")
    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_web_graph()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "What is the NUST Fall 2026 admission deadline?"}]}
    )
    print(result["messages"][-1].content)
