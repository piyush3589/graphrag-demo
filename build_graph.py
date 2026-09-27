"""
build_graph.py
Reads source documents, extracts (entity, relation, entity) triples using an LLM,
and builds a NetworkX knowledge graph. Each entity node stores which source doc(s)
mention it, so retrieval can pull back the original text for grounding.

Run: python build_graph.py
Output: graph.json  (node-link format, loaded later by graph_retriever.py)
"""

import os
import json
import glob
import networkx as nx
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY valid JSON, no markdown fences, in this exact shape:
{{
  "entities": ["Entity Name", ...],
  "relations": [
    {{"source": "Entity A", "relation": "short verb phrase", "target": "Entity B"}}
  ]
}}
Keep entity names short and consistent (e.g. "Vector Database", not "a vector database").

TEXT:
{text}
"""


def load_docs(folder: str) -> dict:
    docs = {}
    for path in glob.glob(os.path.join(folder, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs[os.path.basename(path)] = f.read()
    return docs


def extract_triples(llm: ChatGroq, text: str) -> dict:
    resp = llm.invoke([HumanMessage(content=EXTRACTION_PROMPT.format(text=text))])
    content = resp.content.strip()
    # strip accidental code fences
    if content.startswith("```"):
        content = content.strip("`")
        content = content.split("\n", 1)[1] if "\n" in content else content
        content = content.rsplit("```", 1)[0]
    return json.loads(content)


def build_graph(docs: dict, llm: ChatGroq) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for doc_id, text in docs.items():
        data = extract_triples(llm, text)
        for entity in data.get("entities", []):
            if g.has_node(entity):
                g.nodes[entity]["source_docs"].add(doc_id)
            else:
                g.add_node(entity, source_docs={doc_id})
        for rel in data.get("relations", []):
            src, tgt = rel["source"], rel["target"]
            # relations sometimes name an entity slightly differently than the
            # entities list for the same doc — make sure both endpoints still
            # carry this doc as a source, or their text becomes unreachable later.
            for entity in (src, tgt):
                if g.has_node(entity):
                    g.nodes[entity].setdefault("source_docs", set()).add(doc_id)
                else:
                    g.add_node(entity, source_docs={doc_id})
            g.add_edge(src, tgt, relation=rel["relation"], source_doc=doc_id)
    return g


def save_graph(g: nx.MultiDiGraph, docs: dict, out_path: str = "graph.json"):
    data = nx.node_link_data(g)
    # sets aren't JSON serializable; also guard nodes that were only ever
    # created implicitly via an edge (e.g. relation mentioned an entity name
    # slightly different from the extracted entities list) and never got
    # source_docs set explicitly.
    for node in data["nodes"]:
        node["source_docs"] = list(node.get("source_docs", []))
    payload = {"graph": data, "docs": docs}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    # Groq's JSON mode ("response_format": json_object) makes extraction reliable —
    # without it, smaller open models sometimes wrap JSON in extra prose.
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=2048,
        reasoning_effort="low",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    docs = load_docs("sample_docs")
    print(f"Loaded {len(docs)} documents")
    graph = build_graph(docs, llm)
    print(f"Graph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations")
    save_graph(graph, docs)
    print("Saved to graph.json")
