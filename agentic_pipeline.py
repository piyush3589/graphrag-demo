"""
agentic_pipeline.py
A small LangGraph wrapper around graph_retriever.graph_retrieve().
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from graph_retriever import graph_retrieve

MAX_ATTEMPTS = 3


class AgentState(TypedDict):
    query: str
    hops: int
    attempts: int
    seed_entities: list
    touched_nodes: list
    edges: list
    hop_levels: dict
    context: str
    confident: bool
    answer: str


def search_node(state: AgentState, llm: ChatGroq) -> AgentState:
    result = graph_retrieve(llm, state["query"], hops=state["hops"])
    state["seed_entities"] = result["seed_entities"]
    state["touched_nodes"] = result["touched_nodes"]
    state["edges"] = result["edges"]
    state["hop_levels"] = result["hop_levels"]
    state["context"] = result["context"]
    state["attempts"] += 1
    return state


def evaluate_node(state: AgentState) -> AgentState:
    state["confident"] = bool(state["seed_entities"]) and bool(state["context"].strip())
    return state


def route_after_evaluate(state: AgentState) -> str:
    if state["confident"] or state["attempts"] >= MAX_ATTEMPTS:
        return "answer"
    return "retry"


def retry_node(state: AgentState) -> AgentState:
    state["hops"] += 1
    return state


def answer_node(state: AgentState, llm: ChatGroq) -> AgentState:
    if not state["context"].strip():
        state["answer"] = (
            "I couldn't find anything in these documents connected to that question. "
            "This isn't covered by the current knowledge graph."
        )
        return state

    resp = llm.invoke(
        [
            SystemMessage(
                content="Answer using ONLY the provided context. Be concise and cite which "
                "concepts you connected to form the answer."
            ),
            HumanMessage(content=f"Context:\n{state['context']}\n\nQuestion: {state['query']}"),
        ]
    )
    state["answer"] = resp.content
    return state


def build_agent(llm: ChatGroq):
    graph = StateGraph(AgentState)
    graph.add_node("search", lambda s: search_node(s, llm))
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("retry", retry_node)
    graph.add_node("generate_answer", lambda s: answer_node(s, llm))

    graph.set_entry_point("search")
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_evaluate, {"answer": "generate_answer", "retry": "retry"}
    )
    graph.add_edge("retry", "search")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_agent(llm: ChatGroq, query: str, initial_hops: int = 1) -> AgentState:
    agent = build_agent(llm)
    initial_state: AgentState = {
        "query": query,
        "hops": initial_hops,
        "attempts": 0,
        "seed_entities": [],
        "touched_nodes": [],
        "edges": [],
        "hop_levels": {},
        "context": "",
        "confident": False,
        "answer": "",
    }
    return agent.invoke(initial_state)