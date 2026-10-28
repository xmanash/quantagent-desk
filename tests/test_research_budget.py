"""Research is skipped for numeric books and clipped when documents are needed."""

import os

os.environ.setdefault("QUANTAGENT_SKIP_EMBEDDINGS", "1")

from quantagent_desk.agents.graph import research_plan, retrieve_node
from quantagent_desk.rag.retriever import get_retriever
from quantagent_desk.utils.tokens import estimate_tokens


def test_numeric_queries_skip_research():
    assert research_plan("risk", "Show me firm-wide VaR and ES") is None
    assert research_plan("limits", "Show limit utilization") is None
    assert research_plan("stress", "Run the 2008 GFC stress") is None
    assert research_plan("stress", "Generate an AI bubble unwind stress") is None
    assert research_plan("rag", "What is Expected Shortfall?") is not None


def test_retrieve_node_records_zero_tokens_when_skipped():
    state = {"query": "Show me firm-wide VaR and ES", "intent": "risk", "citations": []}
    out = retrieve_node(state)
    assert out["research_skipped"] is True
    assert out["retrieved"] == []
    assert out["token_use"]["research"] == 0


def test_research_chunks_honor_the_token_budget():
    docs = get_retriever().retrieve("What is Expected Shortfall?", top_k=4, max_tokens=40)
    assert docs
    assert sum(estimate_tokens(doc["text"]) for doc in docs) <= 40
