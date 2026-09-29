"""
Run with:  pytest -v
No API key needed: a fake LLM stands in for Groq.
"""

from types import SimpleNamespace

import networkx as nx
import pytest

from graph_retriever import get_subgraph_context, match_query_entities
from llm_utils import LLMCallError, safe_invoke


# ---------- helpers ----------

class FakeLLM:
    """Returns a fixed reply, like a model would."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, messages):
        return SimpleNamespace(content=self.reply)


class FlakyLLM:
    """Fails `failures` times, then succeeds. Optionally with an HTTP-style status code."""

    def __init__(self, failures: int, status_code=None):
        self.failures = failures
        self.status_code = status_code
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            err = RuntimeError("boom")
            if self.status_code is not None:
                err.status_code = self.status_code
            raise err
        return SimpleNamespace(content="ok")


@pytest.fixture
def chain_graph():
    """Counterparty -> Credit Risk -> Basel III, each entity from a different doc."""
    g = nx.MultiDiGraph()
    g.add_node("Counterparty", source_docs={"doc1.md"})
    g.add_node("Credit Risk", source_docs={"doc2.md"})
    g.add_node("Basel III", source_docs={"doc3.md"})
    g.add_edge("Counterparty", "Credit Risk", relation="drives", source_doc="doc1.md")
    g.add_edge("Credit Risk", "Basel III", relation="feeds capital rules in", source_doc="doc2.md")
    docs = {"doc1.md": "About counterparties.", "doc2.md": "About credit risk.", "doc3.md": "About Basel III."}
    return g, docs


# ---------- retrieval: graph traversal ----------

def test_hop_levels_grow_with_depth(chain_graph):
    g, docs = chain_graph
    _, _, _, levels = get_subgraph_context(g, docs, ["Counterparty"], hops=2)
    assert levels == {"Counterparty": 0, "Credit Risk": 1, "Basel III": 2}


def test_one_hop_does_not_reach_two_hops_away(chain_graph):
    g, docs = chain_graph
    _, nodes, _, _ = get_subgraph_context(g, docs, ["Counterparty"], hops=1)
    assert "Credit Risk" in nodes
    assert "Basel III" not in nodes


def test_context_only_contains_docs_of_touched_entities(chain_graph):
    g, docs = chain_graph
    context, _, _, _ = get_subgraph_context(g, docs, ["Counterparty"], hops=1)
    assert "About counterparties." in context
    assert "About credit risk." in context
    assert "About Basel III." not in context  # not reached at 1 hop


# ---------- retrieval: entity matching ----------

def test_match_returns_empty_when_model_says_none():
    assert match_query_entities(FakeLLM("NONE"), "weather today?", ["Counterparty"]) == []


def test_match_fixes_casing_and_ignores_unknown_names():
    llm = FakeLLM("counterparty, Totally Unknown Thing")
    matched = match_query_entities(llm, "q", ["Counterparty", "Credit Risk"])
    assert matched == ["Counterparty"]


# ---------- error handling ----------

def test_safe_invoke_retries_transient_failure_then_succeeds():
    llm = FlakyLLM(failures=2)
    sleeps = []
    result = safe_invoke(llm, [], retries=3, sleep=sleeps.append)
    assert result.content == "ok"
    assert llm.calls == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff


def test_safe_invoke_gives_up_after_all_retries():
    llm = FlakyLLM(failures=99)
    with pytest.raises(LLMCallError, match="after 3 attempts"):
        safe_invoke(llm, [], retries=3, sleep=lambda s: None)
    assert llm.calls == 3


def test_safe_invoke_does_not_retry_permanent_errors():
    llm = FlakyLLM(failures=99, status_code=404)  # e.g. model not found
    with pytest.raises(LLMCallError, match="non-retryable"):
        safe_invoke(llm, [], retries=3, sleep=lambda s: None)
    assert llm.calls == 1
