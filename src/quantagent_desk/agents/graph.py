"""
LangGraph multi-agent for QuantAgent-Desk.

Nodes:
  router      → classify intent
  retrieve    → RAG over methodology / scenario docs
  risk_tool   → MarketRiskEngine
  stress_tool → StressEngine
  critic      → basic grounding / policy checks
  respond     → format final answer
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

from quantagent_desk.engines import LimitsEngine, MarketRiskEngine, PortfolioEngine, StressEngine
from quantagent_desk.engines.generative_stress import GenerativeStressAgent
from quantagent_desk.guardrails.validator import get_guardrails
from quantagent_desk.memory.manager import get_memory
from quantagent_desk.models.core import Portfolio, QueryResponse
from quantagent_desk.rag.retriever import get_retriever
from quantagent_desk.utils.tokens import (
    RAG_RESEARCH_CHUNKS,
    RAG_RESEARCH_TOKENS,
    STREAM_CHUNK_TOKENS,
    clip_text,
    estimate_tokens,
    iter_token_chunks,
)
from quantagent_desk.utils.tracing import span

# Engines (shared)
_portfolio = PortfolioEngine()
_market = MarketRiskEngine()
_stress = StressEngine()
_limits = LimitsEngine()

# Ensure synthetic data exists
if not _portfolio.get("FIRM"):
    _portfolio.create_synthetic_firm()


class AgentState(TypedDict, total=False):
    query: str
    session_id: str
    portfolio_id: str
    intent: str
    retrieved: List[Dict[str, Any]]
    risk_snapshot: Optional[Dict[str, Any]]
    stress_result: Optional[Dict[str, Any]]
    citations: List[str]
    warnings: List[str]
    calculation_ids: List[str]
    answer: str
    history: str
    research_skipped: bool
    research_tokens: int
    do_research: bool
    token_use: Dict[str, int]
    _blocked: bool
    _grounded: bool


def guardrail_node(state: AgentState) -> AgentState:
    """Entry guardrail node checking if query is allowed."""
    with span("node_guardrail", query=state.get("query", "")[:60]) as sp:
        gr = get_guardrails()
        incoming = gr.check_input(state.get("query", ""))
        if not incoming.allowed:
            state["answer"] = incoming.message
            state["warnings"] = ["Blocked by input policy"]
            state["_blocked"] = True
            state["_grounded"] = True
        else:
            state["_blocked"] = False
        sp.finish(output={"blocked": state["_blocked"]})
    return state


def router_node(state: AgentState) -> AgentState:
    with span("node_router") as sp:
        import re
        q = state["query"].lower()

        def has_word(token: str) -> bool:
            return bool(re.search(rf"\b{re.escape(token)}\b", q))

        def has_phrase(*phrases) -> bool:
            return any(p in q for p in phrases)

        # Order: specific intents first
        if has_phrase("stress", "scenario", "crisis", "shock", "2008", "covid", "gfc", "rates spike", "pandemic"):
            state["intent"] = "stress"
        elif has_phrase("limit", "breach", "warning") or has_word("utilization"):
            state["intent"] = "limits"
        elif has_phrase("value at risk", "expected shortfall", "risk summary", "risk snapshot") or has_word("var") or has_word("es"):
            # if the user is asking for definition, prefer RAG
            if has_phrase("what is", "explain", "definition", "how does", "methodology"):
                state["intent"] = "rag"
            else:
                state["intent"] = "risk"
        elif has_phrase("how", "what is", "explain", "methodology", "definition"):
            state["intent"] = "rag"
        else:
            state["intent"] = "general"
        sp.finish(output={"intent": state["intent"]})
    return state


def planner_node(state: AgentState) -> AgentState:
    """Choose the tool and whether research is worth any tokens. No model call."""
    with span("node_planner", intent=state.get("intent", "general")) as sp:
        plan = research_plan(state.get("intent", ""), state.get("query", ""))
        token_use = dict(state.get("token_use") or {})
        if plan is None:
            state["do_research"] = False
            state["research_skipped"] = True
            state["research_tokens"] = 0
            state["retrieved"] = []
            token_use["research"] = 0
        else:
            state["do_research"] = True
            state["research_skipped"] = False
        state["token_use"] = token_use
        sp.finish(output={"research": bool(state.get("do_research"))})
    return state


def _use_generative(query: str) -> bool:
    q = query.lower()
    gen_keywords = (
        "generate", "ai bubble", "tech crash", "stagflation", "china", "em crisis",
        "liquidity", "funding", "geopolitical", "oil shock", "narrative", "custom stress",
    )
    hist_keywords = ("2008", "gfc", "covid", "2020", "2022 rates", "rates spike", "pandemic")
    return any(k in q for k in gen_keywords) and not any(k in q for k in hist_keywords)


def research_plan(intent: str, _query: str = "") -> Optional[tuple]:
    """Token budget for document research. None means skip retrieval.

    Numeric books and shock building do not need methodology pages.
    Only explanation questions spend a research budget.
    """
    if intent == "rag":
        return RAG_RESEARCH_CHUNKS, RAG_RESEARCH_TOKENS
    return None


def retrieve_node(state: AgentState) -> AgentState:
    with span("node_retrieve", query=state["query"][:60]) as sp:
        plan = research_plan(state.get("intent", ""), state.get("query", ""))
        token_use = dict(state.get("token_use") or {})
        if plan is None:
            state["retrieved"] = []
            state["research_skipped"] = True
            state["research_tokens"] = 0
            token_use["research"] = 0
            state["token_use"] = token_use
            sp.finish(output={"docs_count": 0, "skipped": True})
            return state

        top_k, max_tokens = plan
        docs = get_retriever().retrieve(state["query"], top_k=top_k, max_tokens=max_tokens)
        state["retrieved"] = docs
        state["research_skipped"] = False
        used = sum(int(d.get("tokens") or estimate_tokens(d.get("text", ""))) for d in docs)
        state["research_tokens"] = used
        token_use["research"] = used
        state["token_use"] = token_use
        citations = state.get("citations", [])
        for doc in docs:
            src = doc.get("source", "doc")
            if src not in citations:
                citations.append(src)
        state["citations"] = citations
        sp.finish(output={"docs_count": len(docs), "research_tokens": used, "citations": citations})
    return state


def risk_tool_node(state: AgentState) -> AgentState:
    pid = state.get("portfolio_id") or "FIRM"
    port = _portfolio.get(pid)
    if not port:
        state.setdefault("warnings", []).append(f"Portfolio {pid} not found")
        return state

    flat = _portfolio.flatten_positions(pid)
    temp = Portfolio(portfolio_id=pid, name=port.name, positions=flat)
    snap = _market.from_portfolio_positions(temp)
    breaches = _limits.check(snap, pid)

    state["risk_snapshot"] = snap.model_dump()
    state.setdefault("calculation_ids", []).append(snap.calculation_id)

    answer_parts = [
        f"**Risk Snapshot – {port.name}** (`{pid}`)",
        f"- VaR 95%: **${snap.var_95:,.0f}**",
        f"- VaR 99%: **${snap.var_99:,.0f}**",
        f"- ES 97.5%: **${snap.es_97_5:,.0f}**",
        f"- Methodology: {snap.methodology}",
        f"- Calculation ID: `{snap.calculation_id}`",
    ]
    if breaches:
        answer_parts.append("\n**Limit Status:**")
        for b in breaches:
            util = (b.current_value / b.limit_value) if b.limit_value else 0.0
            answer_parts.append(
                f"- {b.severity}: {b.limit_type} ${b.current_value:,.0f} ({util:.0%})"
            )
    else:
        util = snap.limit_utilization
        if util:
            answer_parts.append("\n**Utilization:** " + ", ".join(f"{k} {v:.0%}" for k, v in util.items()))

    if state.get("retrieved"):
        answer_parts.append("\n**Methodology notes:**")
        for doc in state["retrieved"][:1]:
            answer_parts.append(f"> {doc['text']} _(source: {doc['source']})_")

    state["answer"] = "\n".join(answer_parts)

    mem = get_memory()
    mem.add_episode("calculation", f"Risk snapshot for {pid}", {"var_95": snap.var_95})
    return state


def stress_tool_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    pid = state.get("portfolio_id") or "FIRM"
    port = _portfolio.get(pid)
    flat = _portfolio.flatten_positions(pid)
    temp = Portfolio(portfolio_id=pid, name=port.name if port else pid, positions=flat)
    base = _market.from_portfolio_positions(temp)

    use_gen = _use_generative(q)

    if use_gen:
        agent = GenerativeStressAgent()
        build = agent.generate(state["query"])
        scenario, gwarnings = build.scenario, build.warnings
        token_use = dict(state.get("token_use") or {})
        token_use["prompt"] = build.prompt_tokens
        token_use["completion"] = build.completion_tokens
        state["token_use"] = token_use
        state.setdefault("warnings", []).extend(gwarnings)
        if scenario is None:
            state["answer"] = gwarnings[0] if gwarnings else "Stress blocked by policy."
            return state
    else:
        if "2008" in q or "gfc" in q:
            key = "2008_GFC"
        elif "covid" in q or "2020" in q or "pandemic" in q:
            key = "COVID_2020"
        elif "rates" in q or "2022" in q:
            key = "RATES_SPIKE_2022"
        else:
            key = "2008_GFC"
        scenario = _stress.get_historical(key)
        if not scenario:
            state.setdefault("warnings", []).append("Scenario not found")
            state["answer"] = "Scenario not found."
            return state

    result = _stress.apply_stress(temp, scenario, base_var=base.var_95)
    state["stress_result"] = result.model_dump()
    state.setdefault("calculation_ids", []).append(result.calculation_id)
    state.setdefault("citations", []).extend(list(scenario.grounded_sources))

    lines = [
        f"**Stress Result: {scenario.name}** ({scenario.type})",
        scenario.description,
        "",
        f"- Estimated P&L Impact: **${result.pnl_impact:,.0f}**",
        f"- VaR Impact: **${result.var_impact:,.0f}**",
        f"- Calculation ID: `{result.calculation_id}`",
    ]
    if scenario.shocks:
        lines.append("")
        lines.append("**Shocks applied:**")
        for sh in scenario.shocks:
            lines.append(f"- {sh.factor}: {sh.shock_type} {sh.value}")
    if result.breakdown:
        lines.append("")
        lines.append("**Top instrument impacts:**")
        for instr, imp in sorted(result.breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            lines.append(f"- {instr}: ${imp:,.0f}")
    if state.get("retrieved"):
        lines.append("")
        lines.append("**Scenario context:**")
        for doc in state["retrieved"][:1]:
            lines.append(f"> {doc['text']} (source: {doc['source']})")

    state["answer"] = "\n".join(lines)
    get_memory().add_episode("stress", f"Ran {scenario.name} on {pid}", {"pnl_impact": result.pnl_impact})
    return state


def limits_node(state: AgentState) -> AgentState:
    state = risk_tool_node(state)
    snap = state.get("risk_snapshot")
    if snap:
        util = snap.get("limit_utilization", {})
        lines = ["**Current Limit Utilization**"]
        for k, v in util.items():
            status = "BREACH" if v >= 1.0 else ("WARNING" if v >= 0.8 else "OK")
            lines.append(f"- {k}: {v:.1%} ({status})")
        state["answer"] = "\n".join(lines)
    return state


def rag_only_node(state: AgentState) -> AgentState:
    docs = state.get("retrieved") or []
    if not docs:
        state["answer"] = "I could not find relevant methodology documents for that question."
        return state
    parts = ["**From methodology documentation:**\n"]
    for d in docs[:2]:
        parts.append(f"### {d['source']}\n{d['text']}\n")
    state["answer"] = "\n".join(parts)
    return state


def general_node(state: AgentState) -> AgentState:
    recent = get_memory().recent_episodes(limit=3)
    ep_text = "\n".join(f"- [{e['kind']}] {e['summary']}" for e in recent) if recent else "None yet."
    ep_text = clip_text(ep_text, 80)

    state["answer"] = (
        "I am **QuantAgent-Desk**, your multi-agent risk assistant.\n\n"
        "I can help with:\n"
        "- Current **VaR / ES / risk summary**\n"
        "- Running **historical or hybrid stress** scenarios (2008, COVID, 2022 rates…)\n"
        "- **Limit utilization** and breaches\n"
        "- Questions about **methodology** (VaR, ES, stress definitions)\n\n"
        f"**Recent activity:**\n{ep_text}\n\n"
        "Try: *“Show firm-wide VaR”* or *“Run 2008 stress on Rates desk”*"
    )
    return state


def critic_node(state: AgentState) -> AgentState:
    """Numeric grounding + policy critic via Guardrails."""
    if state.get("_blocked"):
        return state
    with span("node_critic", intent=state.get("intent"), calc_ids=state.get("calculation_ids")) as sp:
        gr = get_guardrails()
        research = [d.get("text", "") for d in (state.get("retrieved") or []) if d.get("text")]
        answer, warnings, grounded = gr.ground_response(
            answer=state.get("answer", ""),
            calculation_ids=state.get("calculation_ids", []),
            citations=state.get("citations", []),
            intent=state.get("intent", "general"),
            snapshot=state.get("risk_snapshot"),
            stress_result=state.get("stress_result"),
            research=research,
        )
        state["answer"] = answer
        existing = state.get("warnings", [])
        state["warnings"] = existing + [w for w in warnings if w not in existing]
        state["_grounded"] = grounded
        token_use = dict(state.get("token_use") or {})
        token_use["answer"] = estimate_tokens(answer)
        token_use["critic"] = 0 if grounded else estimate_tokens("Ungrounded figure removed.")
        state["token_use"] = token_use
        sp.finish(output={"grounded": grounded, "tokens": token_use})
    return state


def route_after_guardrail(state: AgentState) -> str:
    """Route blocked queries straight to critic -> END; otherwise to router."""
    if state.get("_blocked"):
        return "critic"
    return "router"


def build_langgraph(checkpointer: Optional[Any] = None):
    """Build and return a compiled LangGraph StateGraph with checkpointer."""
    try:
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.memory import MemorySaver

        graph = StateGraph(AgentState)
        graph.add_node("guardrail", guardrail_node)
        graph.add_node("router", router_node)
        graph.add_node("planner", planner_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("risk", risk_tool_node)
        graph.add_node("stress", stress_tool_node)
        graph.add_node("limits", limits_node)
        graph.add_node("rag", rag_only_node)
        graph.add_node("general", general_node)
        graph.add_node("critic", critic_node)

        graph.add_edge(START, "guardrail")
        graph.add_conditional_edges(
            "guardrail",
            route_after_guardrail,
            {
                "critic": "critic",
                "router": "router",
            },
        )
        graph.add_edge("router", "planner")

        def route_after_planner(state: AgentState) -> str:
            if state.get("do_research"):
                return "retrieve"
            return state.get("intent", "general")

        graph.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "retrieve": "retrieve",
                "risk": "risk",
                "stress": "stress",
                "limits": "limits",
                "rag": "rag",
                "general": "general",
            },
        )

        def route_after_retrieve(state: AgentState) -> str:
            return state.get("intent", "general")

        graph.add_conditional_edges(
            "retrieve",
            route_after_retrieve,
            {
                "risk": "risk",
                "stress": "stress",
                "limits": "limits",
                "rag": "rag",
                "general": "general",
            },
        )
        for node in ["risk", "stress", "limits", "rag", "general"]:
            graph.add_edge(node, "critic")
        graph.add_edge("critic", END)

        cp = checkpointer if checkpointer is not None else MemorySaver()
        return graph.compile(checkpointer=cp)
    except Exception as e:
        import logging
        logging.getLogger("quantagent_desk.agents.graph").warning("Failed to compile LangGraph: %s", e)
        return None


_compiled_graph = None


def _to_response(state: AgentState, session_id: str) -> QueryResponse:
    from quantagent_desk.models.core import RiskSnapshot, StressResult

    snap_obj = None
    if state.get("risk_snapshot"):
        try:
            snap_obj = RiskSnapshot(**state["risk_snapshot"])
        except Exception:
            pass
    stress_obj = None
    if state.get("stress_result"):
        try:
            stress_obj = StressResult(**state["stress_result"])
        except Exception:
            pass
    research = [d.get("text", "") for d in (state.get("retrieved") or []) if d.get("text")]
    return QueryResponse(
        answer=state.get("answer", ""),
        session_id=session_id,
        calculation_ids=state.get("calculation_ids", []),
        citations=state.get("citations", []),
        charts=[],
        risk_snapshot=snap_obj,
        stress_result=stress_obj,
        grounded=state.get("_grounded", len(state.get("warnings", [])) == 0),
        warnings=state.get("warnings", []),
        research=research,
        token_use=dict(state.get("token_use") or {}),
    )


def get_compiled_graph():
    """Return cached singleton compiled StateGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_langgraph()
    return _compiled_graph


def build_and_run(query: str, session_id: str = None, portfolio_id: str = "FIRM") -> QueryResponse:
    """Execute the compiled LangGraph agent pipeline with guardrails + tracing."""
    session_id = session_id or str(uuid4())
    mem = get_memory()

    state: AgentState = {
        "query": query,
        "session_id": session_id,
        "portfolio_id": portfolio_id,
        "intent": "general",
        "retrieved": [],
        "risk_snapshot": None,
        "stress_result": None,
        "citations": [],
        "warnings": [],
        "calculation_ids": [],
        "answer": "",
        "history": "",
        "_blocked": False,
        "_grounded": True,
    }

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}

    with span("agent_pipeline", query=query[:80], portfolio_id=portfolio_id) as sp:
        if graph is not None:
            final_state = graph.invoke(state, config=config)
        else:
            final_state = _run_inline(state)
        tokens = final_state.get("token_use") or {}
        sp.finish(output={
            "intent": final_state.get("intent"),
            "calc_count": len(final_state.get("calculation_ids") or []),
            "tokens": tokens,
        })

    mem.add_turn(session_id, "user", query)
    mem.add_turn(
        session_id,
        "assistant",
        final_state.get("answer", ""),
        meta={"intent": final_state.get("intent"), "calc_ids": final_state.get("calculation_ids", [])},
    )

    return _to_response(final_state, session_id)


def _run_inline(state: AgentState) -> AgentState:
    """Same planner path when the compiled graph cannot be built."""
    state = guardrail_node(state)
    if state.get("_blocked"):
        return critic_node(state)
    state = router_node(state)
    state = planner_node(state)
    if state.get("do_research"):
        state = retrieve_node(state)
    intent = state.get("intent")
    if intent == "risk":
        state = risk_tool_node(state)
    elif intent == "stress":
        state = stress_tool_node(state)
    elif intent == "limits":
        state = limits_node(state)
    elif intent == "rag":
        state = rag_only_node(state)
    else:
        state = general_node(state)
    return critic_node(state)


def _stream_inline(state: AgentState):
    """Node events for the inline path. The last event carries the state."""
    state = guardrail_node(state)
    yield {
        "event": "node_update",
        "node": "guardrail",
        "status": "Blocked by input policy" if state.get("_blocked") else "Input validated by guardrails",
        "data": {"blocked": state.get("_blocked", False)},
    }
    if not state.get("_blocked"):
        state = router_node(state)
        yield {
            "event": "node_update",
            "node": "router",
            "status": f"Intent classified: {str(state.get('intent', 'general')).upper()}",
            "data": {"intent": state.get("intent")},
        }
        state = planner_node(state)
        skipped = bool(state.get("research_skipped"))
        yield {
            "event": "node_update",
            "node": "planner",
            "status": f"Plan {str(state.get('intent', 'general')).upper()}: research {'off' if skipped else 'on'}",
            "data": {"research": not skipped},
        }
        if state.get("do_research"):
            state = retrieve_node(state)
            used = state.get("research_tokens", 0)
            count = len(state.get("retrieved") or [])
            yield {
                "event": "node_update",
                "node": "retrieve",
                "status": f"Retrieved {count} methodology documents ({used} tok)",
                "data": {"count": count, "research_tokens": used},
            }
        intent = state.get("intent")
        if intent == "risk":
            state = risk_tool_node(state)
        elif intent == "stress":
            state = stress_tool_node(state)
        elif intent == "limits":
            state = limits_node(state)
        elif intent == "rag":
            state = rag_only_node(state)
        else:
            state = general_node(state)
        yield {
            "event": "node_update",
            "node": intent or "general",
            "status": f"Computed {intent}",
            "data": {},
        }
    state = critic_node(state)
    yield {
        "event": "node_update",
        "node": "critic",
        "status": "Critic verified numeric grounding & policy",
        "data": {"grounded": state.get("_grounded", True)},
    }
    yield {"event": "state", "state": state}


def stream_graph(query: str, session_id: Optional[str] = None, portfolio_id: str = "FIRM"):
    """Stream LangGraph node execution events, token chunks, and final QueryResponse."""
    session_id = session_id or str(uuid4())
    mem = get_memory()

    state: AgentState = {
        "query": query,
        "session_id": session_id,
        "portfolio_id": portfolio_id,
        "intent": "general",
        "retrieved": [],
        "risk_snapshot": None,
        "stress_result": None,
        "citations": [],
        "warnings": [],
        "calculation_ids": [],
        "answer": "",
        "history": "",
        "_blocked": False,
        "_grounded": True,
    }

    yield {
        "event": "start",
        "session_id": session_id,
        "portfolio_id": portfolio_id,
    }

    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    accumulated_state: Dict[str, Any] = dict(state)

    with span("agent_pipeline_stream", query=query[:80], portfolio_id=portfolio_id) as sp:
        if graph is not None:
            for step in graph.stream(state, config=config, stream_mode="updates"):
                for node_name, node_output in step.items():
                    accumulated_state.update(node_output)
                    status_msg = ""
                    data_payload = {}
                    if node_name == "guardrail":
                        blocked = node_output.get("_blocked", False)
                        status_msg = "Blocked by input policy" if blocked else "Input validated by guardrails"
                        data_payload = {"blocked": blocked}
                    elif node_name == "router":
                        intent = node_output.get("intent", "general")
                        status_msg = f"Intent classified: {intent.upper()}"
                        data_payload = {"intent": intent}
                    elif node_name == "planner":
                        intent = node_output.get("intent", "general")
                        if node_output.get("research_skipped"):
                            status_msg = f"Plan {str(intent).upper()}: research off"
                        else:
                            status_msg = f"Plan {str(intent).upper()}: research on"
                        data_payload = {"research": not node_output.get("research_skipped", False)}
                    elif node_name == "retrieve":
                        count = len(node_output.get("retrieved", []))
                        citations = node_output.get("citations", [])
                        if node_output.get("research_skipped"):
                            status_msg = "Research skipped to save tokens"
                        else:
                            used = node_output.get("research_tokens", 0)
                            status_msg = f"Retrieved {count} methodology documents ({used} tok)"
                        data_payload = {"count": count, "citations": citations, "research_tokens": node_output.get("research_tokens", 0)}
                    elif node_name == "risk":
                        calcs = node_output.get("calculation_ids", [])
                        status_msg = f"Market risk snapshot computed ({calcs[-1] if calcs else 'OK'})"
                        data_payload = {"calculation_ids": calcs}
                    elif node_name == "stress":
                        calcs = node_output.get("calculation_ids", [])
                        status_msg = "Stress scenario impact evaluated"
                        data_payload = {"calculation_ids": calcs}
                    elif node_name == "limits":
                        status_msg = "Limit utilization checked"
                    elif node_name == "rag":
                        status_msg = "Methodology documentation synthesized"
                    elif node_name == "general":
                        status_msg = "Assistant overview prepared"
                    elif node_name == "critic":
                        grounded = node_output.get("_grounded", True)
                        status_msg = "Critic verified numeric grounding & policy"
                        data_payload = {"grounded": grounded}

                    yield {
                        "event": "node_update",
                        "node": node_name,
                        "status": status_msg,
                        "data": data_payload,
                    }
        else:
            for event in _stream_inline(accumulated_state):
                if event.get("event") == "state":
                    accumulated_state = event["state"]
                else:
                    yield event

        tokens = accumulated_state.get("token_use") or {}
        sp.finish(output={
            "intent": accumulated_state.get("intent"),
            "calc_count": len(accumulated_state.get("calculation_ids") or []),
            "tokens": tokens,
        })

    answer = accumulated_state.get("answer", "")
    for chunk in iter_token_chunks(answer, STREAM_CHUNK_TOKENS):
        yield {"event": "token", "chunk": chunk}

    # Record turns in session memory
    mem.add_turn(session_id, "user", query)
    mem.add_turn(
        session_id,
        "assistant",
        answer,
        meta={"intent": accumulated_state.get("intent"), "calc_ids": accumulated_state.get("calculation_ids", [])},
    )

    final_resp = _to_response(accumulated_state, session_id)

    yield {
        "event": "final",
        "response": final_resp.model_dump(),
    }
