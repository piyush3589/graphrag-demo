import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from graph_retriever import graph_retrieve
from agentic_pipeline import run_agent
from llm_utils import safe_invoke, LLMCallError
from dotenv import load_dotenv

load_dotenv()

ANSWER_SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. Be concise and cite which "
    "concepts you connected to form the answer."
)


def run_query(llm: ChatGroq, query: str, agentic: bool):
    """Returns (retrieval_result, answer_text, agent_info_or_None)."""
    if agentic:
        state = run_agent(llm, query, initial_hops=1)
        result = {
            "seed_entities": state["seed_entities"],
            "touched_nodes": state["touched_nodes"],
            "edges": state["edges"],
            "hop_levels": state["hop_levels"],
            "context": state["context"],
        }
        info = f"Agent took {state['attempts']} attempt(s), ending at hop depth {state['hops']}."
        return result, state["answer"], info

    result = graph_retrieve(llm, query, hops=2)
    if not result["context"].strip():
        return result, "Nothing in these documents is connected to that question.", None
    resp = safe_invoke(
        llm,
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{result['context']}\n\nQuestion: {query}"),
        ],
    )
    return result, resp.content, None


st.set_page_config(page_title="GraphRAG Demo", layout="wide")
st.title("GraphRAG Q&A Demo")
st.caption(
    "Answers multi-hop questions by walking a knowledge graph built from the docs, "
    "not just matching similar-looking text chunks (standard RAG)."
)

agentic_mode = st.checkbox(
    "Agentic mode — auto-retry with a deeper graph walk if the first attempt finds too little",
    value=False,
)

query = st.text_input(
    "Ask a question that spans multiple documents",
    placeholder="How does a Counterparty's Risk Rating end up affecting the bank's Capital requirements?",
)

if st.button("Ask") and query:
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

    try:
        with st.spinner("Walking the graph..."):
            result, answer_text, agent_info = run_query(llm, query, agentic_mode)
    except FileNotFoundError:
        st.error("graph.json not found. Run `python build_graph.py` first to build the knowledge graph.")
        st.stop()
    except LLMCallError as err:
        st.error(f"The language model call failed: {err}")
        st.stop()

    if agent_info:
        st.info(agent_info)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Answer")
        st.write(answer_text)

    with col2:
        st.subheader("Graph path used")
        st.write("**Seed entities from query (hop 0):**", result["seed_entities"])

        max_hop = max(result["hop_levels"].values(), default=0)
        for h in range(1, max_hop + 1):
            entities_at_hop = [e for e, lvl in result["hop_levels"].items() if lvl == h]
            if entities_at_hop:
                st.write(f"**Hop {h}:**", entities_at_hop)

        st.write("**Relationships traversed:**")
        for s, rel, t, hop_num in result["edges"]:
            st.write(f"- (hop {hop_num}) {s} —({rel})→ {t}")

    st.divider()
    with st.expander("Raw retrieved context (grounding)"):
        st.text(result["context"])