"""
graph_retriever.py
Given a query, this:
  1. Asks the LLM which known entities the query touches
  2. Walks the graph up to N hops from those entities (this is the part
     standard vector-similarity RAG cannot do — it has no notion of
     "connected" facts, only "similar-looking" text)
  3. Collects the source doc text for every entity/edge touched
  4. Returns that as grounded context, plus the subgraph path for display
"""

import json
import networkx as nx
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def load_graph(path: str = "graph.json"):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    g = nx.node_link_graph(payload["graph"], directed=True, multigraph=True)
    return g, payload["docs"]


def match_query_entities(llm: ChatGroq, query: str, known_entities: list[str]) -> list[str]:
    prompt = (
        f"Known entities: {known_entities}\n\n"
        f"Question: {query}\n\n"
        "Which of the known entities (exact names only, comma-separated, no explanation) "
        "does this question genuinely relate to? If NONE of them are actually relevant to "
        "the question's topic, respond with exactly: NONE"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = resp.content.strip()
    if raw.upper() == "NONE":
        return []
    named = [e.strip() for e in raw.split(",")]

    # exact match first, then case-insensitive, then substring — the LLM often
    # doesn't return the exact casing/wording of the known entity list.
    matched = []
    lower_known = {e.lower(): e for e in known_entities}
    for n in named:
        if n in known_entities:
            matched.append(n)
        elif n.lower() in lower_known:
            matched.append(lower_known[n.lower()])
        else:
            for known in known_entities:
                if n.lower() in known.lower() or known.lower() in n.lower():
                    matched.append(known)
                    break
    return list(dict.fromkeys(matched))  # dedupe, keep order


def get_subgraph_context(g: nx.MultiDiGraph, docs: dict, seed_entities: list[str], hops: int = 1):
    touched_nodes = set(seed_entities)
    edges_used = []
    hop_levels = {entity: 0 for entity in seed_entities}  # seeds are hop 0

    frontier = set(seed_entities)
    for hop_num in range(1, hops + 1):
        next_frontier = set()
        for node in frontier:
            if node not in g:
                continue
            for _, target, data in g.out_edges(node, data=True):
                edges_used.append((node, data.get("relation", "related to"), target, hop_num))
                next_frontier.add(target)
            for source, _, data in g.in_edges(node, data=True):
                edges_used.append((source, data.get("relation", "related to"), node, hop_num))
                next_frontier.add(source)
        for entity in next_frontier:
            if entity not in hop_levels:
                hop_levels[entity] = hop_num
        touched_nodes |= next_frontier
        frontier = next_frontier

    # collect source docs for every touched node
    doc_ids = set()
    for node in touched_nodes:
        if node in g:
            doc_ids |= set(g.nodes[node].get("source_docs", []))

    context_text = "\n\n".join(f"[{d}]\n{docs[d]}" for d in doc_ids if d in docs)
    return context_text, list(touched_nodes), edges_used, hop_levels


def graph_retrieve(llm: ChatGroq, query: str, hops: int = 1):
    g, docs = load_graph()
    known_entities = list(g.nodes)
    seeds = match_query_entities(llm, query, known_entities)
    context, nodes, edges, hop_levels = get_subgraph_context(g, docs, seeds, hops=hops)
    return {
        "context": context,
        "seed_entities": seeds,
        "touched_nodes": nodes,
        "edges": edges,
        "hop_levels": hop_levels,
    }