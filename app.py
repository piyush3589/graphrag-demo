import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from graph_retriever import graph_retrieve
from agentic_pipeline import run_agent
from dotenv import load_dotenv

load_dotenv()

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

    if agentic_mode:
        with st.spinner("Searching, and retrying deeper if needed..."):
            state = run_agent(llm, query, initial_hops=1)
        result = {
            "seed_entities": state["seed_entities"],
            "touched_nodes": state["touched_nodes"],
            "edges": state["edges"],
            "hop_levels": state["hop_levels"],
            "context": state["context"],
        }
        answer_text = state["answer"]
        st.info(f"Agent took {state['attempts']} attempt(s), ending at hop depth {state['hops']}.")
    else:
        with st.spinner("Walking the graph..."):
            result = graph_retrieve(llm, query, hops=2)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Answer")
        if agentic_mode:
            st.write(answer_text)
        else:
            answer = llm.invoke(
                [
                    SystemMessage(
                        content="Answer using ONLY the provided context. Be concise and cite which "
                        "concepts you connected to form the answer."
                    ),
                    HumanMessage(content=f"Context:\n{result['context']}\n\nQuestion: {query}"),
                ]
            )
            st.write(answer.content)

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