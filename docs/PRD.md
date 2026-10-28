# QuantAgent-Desk – Product Requirements Document (Summary)

Full detailed PRD is available in the conversation history. This file captures the essential points for the repository.

## Vision
 unified risk desk powered by multi-agent AI with RAG, memory, guardrails, tracing and continuous evaluation.

## Core Capabilities (v0.1 delivered)
- Hierarchical portfolio (Firm → Desk → Book)
- Historical / parametric VaR & ES
- Pre-built historical stress scenarios (2008, COVID, 2022 rates)
- Simple generative / hybrid stress hooks, plus an LLM narrative → shock builder (template fallback offline)
- Limit utilization & breach detection
- FastAPI backend
- Streamlit similar to Bloomberg multi-panel terminal (Risk, Limits, Stress, Chat on one screen)
- Synthetic data generator

## Next Iterations
1. Hybrid RAG over methodology docs + past reports
2. Persistent memory (session + entity + episodic)
3. Optional integration points with DerivPricer-AI and CCR-XVA-Monitor

Delivered in the current tree: compiled LangGraph (planner → one tool → critic) with streaming, numeric grounding that drops unmatched figures, and local traces with optional clipped Langfuse or Phoenix export.

## License
MIT
