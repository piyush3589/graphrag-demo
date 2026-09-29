<div align="center">

# 🕸️ GraphRAG Q&A Demo

### Answering questions by *walking connections*, not just matching similar text

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-1C3C3C)](https://www.langchain.com/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Inference-F55036)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

*A working GraphRAG + agentic retrieval pipeline, built and debugged end-to-end,*
*demonstrated on a Risk & Treasury banking domain.*

</div>

---

## 💡 What this actually does

Ask a question like:

> *"Why would a wave of Counterparty defaults also become a Liquidity Risk problem?"*

Standard RAG finds text that **sounds similar** to your question. It has no way to
connect facts that live in *different documents* using *different vocabulary*.

This project builds a **knowledge graph** from the source documents first — entities
and the relationships between them — so it can **walk the connections**:

```
Counterparty → (default triggers) → Credit Risk → (drains) → Liquid Asset Buffers → (managed by) → Treasury Desk
```

No single document states that full chain. Standard vector search would never find it.
Graph traversal does — because it doesn't need the words to match, only the *concepts*
to connect.

---

## ✨ Features

| | |
|---|---|
| 🔗 **Multi-hop graph retrieval** | Walks entity relationships up to N hops, not just top-K similar chunks |
| 🤖 **Agentic mode** | LangGraph-driven agent auto-retries with a deeper search if the first pass finds too little |
| 🚫 **Honest fallback** | Genuinely says "not covered by these documents" instead of forcing a fake answer |
| 🔍 **Transparent retrieval** | The UI shows *exactly* which entities and edges were walked, hop by hop — not a black box |
| ⚡ **Fast, free-tier friendly** | Runs on Groq's hosted inference, no GPU or OpenAI key required |

---

## 🏗️ Architecture

```
sample_docs/*.md
       │
       ▼
┌─────────────────┐     LLM extracts (entity, relation, entity)
│  build_graph.py │ ──► triples from each document
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   graph.json    │  NetworkX MultiDiGraph — entities as nodes,
│  (the graph)    │  relationships as edges, source docs attached
└─────────────────┘
       │
       ▼
┌────────────────────┐   Finds seed entities → walks N hops →
│ graph_retriever.py  │ ─ collects grounded context, tagged by hop level
└────────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌─────────────┐          ┌─────────────────────┐
│   app.py    │          │ agentic_pipeline.py  │
│ Streamlit UI │ ◄─────── │  LangGraph agent:     │
│             │          │  search→evaluate→     │
│             │          │  retry deeper if weak  │
└─────────────┘          └─────────────────────┘
```

| File | Role |
|---|---|
| `build_graph.py` | One-time build step — extracts the knowledge graph from `sample_docs/` |
| `graph_retriever.py` | Core retrieval logic — matches query to entities, walks the graph, tracks hop depth |
| `agentic_pipeline.py` | Optional decision layer (LangGraph) — decides whether to retry with a deeper search |
| `app.py` | Streamlit UI — ask questions, see the answer *and* the graph path that produced it |

**GraphRAG** (the graph itself) is the *knowledge layer*. **LangGraph** (the agent) is
the *control layer* — deciding *when* and *how deep* to search. They're separate tools
solving separate problems, both present in this project.

---

## 🚀 Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key (free tier at console.groq.com)
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Build the knowledge graph (one-time step)
python build_graph.py

# 4. Launch the app
streamlit run app.py
```

### 🧪 Try these questions

**Standard multi-hop:**
> How does a Counterparty's Risk Rating end up affecting the bank's Capital requirements under Basel III?

**Deep, 3+ document chain:**
> Why would a wave of Counterparty defaults also become a Liquidity Risk problem for the Treasury Desk?

**Edge case (test the honest fallback):**
> What's the weather like today?
> *— toggle "Agentic mode" first to watch it retry with deeper searches before giving up gracefully.*

---

## ☁️ Deploying to Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**
2. Push this folder (including `sample_docs/` and `graph.json`) to the Space repo
3. Add `GROQ_API_KEY` under **Settings → Repository secrets**
4. Done — no GPU required, runs entirely on Groq's hosted inference

---

## 🔄 Swap in your own documents

Replace the files in `sample_docs/` with anything — project READMEs, other domain
notes, whatever — and re-run `python build_graph.py`. The pipeline is fully
content-agnostic; the Risk/Treasury framing here is just the demo domain.

---

## 🗣️ Major points

- **Why GraphRAG over standard RAG** — multi-hop reasoning across documents that don't share vocabulary, at the cost of an upfront graph-building pass
- **Where it fits** — genuinely interconnected domains (Risk/Treasury, compliance) benefit most; loosely-related document dumps don't need the extra cost
- **GraphRAG vs. LangGraph** — knowledge layer vs. control layer, not the same thing, both used here for different jobs


</br>

<div align="center">
<sub>Built as a portfolio project — debugged end-to-end, including real issues along the way:
SSL interception, model access limits, LangGraph node-naming constraints, and a
silent prompt-design bug that made the agent's fallback logic never actually fire.</sub>
</div>
