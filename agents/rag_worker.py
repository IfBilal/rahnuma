"""
RAG worker subgraph: query rewrite -> hybrid retrieval (BM25 + pgvector) ->
relevance grading -> answer, with a corrective fallback when the corpus
genuinely doesn't have relevant info. This replaces M1's single-node graph's
naive retrieval with the real agentic RAG pattern from the PRD.

Query rewriting is also what fixes the M1 known limitation: follow-up
questions ("how does that compare to their BBA formula?") get rewritten
into a standalone, retrievable question using conversation history, instead
of being searched verbatim.
"""

from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from fastembed.rerank.cross_encoder import TextCrossEncoder
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from ingestion.store import get_vector_store
import os
from psycopg import connect

from agents.web_worker import build_web_graph

llm = init_chat_model("groq:llama-3.3-70b-versatile")


def load_indexed_chunks() -> list[Document]:
    """Load BM25's documents from pgvector instead of reparsing every PDF.

    The earlier implementation re-read and chunked the raw corpus whenever a
    serving process first received a RAG request. That made the first answer
    slow and vulnerable to a single expensive PDF extraction. pgvector already
    holds the exact post-ingestion text and citation metadata we need, so it is
    the correct source for the in-memory lexical index.
    """
    connection_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://")
    with connect(connection_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = %s
            """,
            ("rahnuma_corpus",),
        )
        rows = cursor.fetchall()

    if not rows:
        raise RuntimeError("The Rahnuma corpus is empty. Run `python -m ingestion.run` first.")
    return [Document(page_content=document, metadata=metadata or {}) for document, metadata in rows]


def build_hybrid_retriever(k: int = 10):
    # retrieve more candidates than we ultimately want — the reranker below
    # narrows these down to the truly best matches, not just the first ones
    # each individual retriever happened to find
    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(load_indexed_chunks())
    bm25_retriever.k = k
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])


# These resources are intentionally initialized lazily. Constructing PGVector
# opens a database connection and loading the cross-encoder can download/load
# an ONNX model. Doing either during an `import api.main` made health checks
# and unrelated endpoints fail whenever Postgres was unavailable.
_retriever = None
_web_graph = None
_reranker = None
RERANK_TOP_N = 10


def get_retrieval_runtime():
    """Create expensive, process-wide RAG resources only when a RAG request arrives."""
    global _retriever, _web_graph, _reranker
    if _retriever is None:
        _retriever = build_hybrid_retriever()
    if _web_graph is None:
        _web_graph = build_web_graph()
    if _reranker is None:
        # A cross-encoder scores query + candidate together. It is more precise
        # than embeddings, so we use it only on the small retrieved candidate set.
        _reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    return _retriever, _web_graph, _reranker


class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    rewritten_query: str
    context: str
    is_relevant: bool


def rewrite_query(state: RAGState) -> dict:
    history = state["messages"]
    if len(history) == 1:
        # first turn, nothing to condense against
        return {"rewritten_query": history[-1].content}

    conversation = "\n".join(f"{m.type}: {m.content}" for m in history)
    prompt = f"""Given this conversation, rewrite the final user question into a standalone
question that makes sense without needing the earlier conversation for context.
Only output the rewritten question, nothing else.

Conversation:
{conversation}"""
    response = llm.invoke(prompt)
    return {"rewritten_query": response.content.strip()}


def retrieve(state: RAGState) -> dict:
    query = state["rewritten_query"]
    retriever, _, reranker = get_retrieval_runtime()
    docs = retriever.invoke(query)

    scores = list(reranker.rerank(query, [doc.page_content for doc in docs]))
    reranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    top_docs = [doc for doc, score in reranked[:RERANK_TOP_N]]

    context = "\n\n".join(
        f"[{doc.metadata.get('university', 'unknown')}, {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in top_docs
    )
    return {"context": context}


def grade(state: RAGState) -> dict:
    prompt = f"""Question: {state['rewritten_query']}

Retrieved context:
{state['context']}

Does this context contain information relevant enough to answer the question?
Answer with exactly one word: "yes" or "no"."""
    response = llm.invoke(prompt)
    is_relevant = "yes" in response.content.strip().lower()
    return {"is_relevant": is_relevant}


def route_after_grade(state: RAGState) -> str:
    return "answer" if state["is_relevant"] else "fallback"


PROMPT_TEMPLATE = """You are answering questions about Pakistani university admissions using ONLY the context below. Footnotes and asterisked notes in the context are valid information — treat them the same as regular text.

Context:
{context}

Question: {question}

Answer using the context above, citing university and page for each claim."""


def answer(state: RAGState) -> dict:
    prompt = PROMPT_TEMPLATE.format(context=state["context"], question=state["rewritten_query"])
    response = llm.invoke(prompt)
    return {"messages": [response]}


def fallback(state: RAGState) -> dict:
    # corrective RAG: corpus retrieval failed the relevance check, so escalate
    # to the live web worker instead — for stale/missing corpus info (e.g.
    # current-cycle deadlines the prospectus doesn't have yet).
    _, web_graph, _ = get_retrieval_runtime()
    result = web_graph.invoke({"messages": [{"role": "user", "content": state["rewritten_query"]}]})
    # overwrite context with what actually grounds the final answer (web
    # results, not the corpus context that just failed the relevance check)
    return {"messages": [result["messages"][-1]], "context": result.get("search_results", "")}


def build_rag_graph(checkpointer=None):
    graph_builder = StateGraph(RAGState)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("grade", grade)
    graph_builder.add_node("answer", answer)
    graph_builder.add_node("fallback", fallback)

    graph_builder.set_entry_point("rewrite_query")
    graph_builder.add_edge("rewrite_query", "retrieve")
    graph_builder.add_edge("retrieve", "grade")
    graph_builder.add_conditional_edges("grade", route_after_grade, {"answer": "answer", "fallback": "fallback"})
    graph_builder.set_finish_point("answer")
    graph_builder.set_finish_point("fallback")

    return graph_builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_rag_graph()

    print("--- Turn 1 ---")
    result = graph.invoke({"messages": [{"role": "user", "content": "What is FAST's merit formula for BS Engineering?"}]})
    print(result["messages"][-1].content)

    print("\n--- Turn 2 (follow-up, tests query rewriting) ---")
    all_messages = result["messages"] + [{"role": "user", "content": "How does that compare to their BBA formula?"}]
    result2 = graph.invoke({"messages": all_messages})
    print(result2["messages"][-1].content)
