"""Lightweight accuracy + RAG-style evaluation suite (no external LLM required)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    name: str
    query: str
    expect_intent: str
    expect_has_calc_id: bool = False
    expect_grounded: bool = True
    expect_keywords: List[str] = field(default_factory=list)


# Accuracy gate used by CI. RAGAS scores are lexical (no judge model) unless
# QUANTAGENT_RAGAS_LEXICAL=0.
MIN_PASS_RATE = 1.0
MIN_RAGAS_SCORE = 0.70
MIN_NUMERIC_GROUNDING = 0.80

# Max 5 realistic scenarios for demo / CI
EVAL_CASES: List[EvalCase] = [
    EvalCase(
        name="firm_var",
        query="Show me firm-wide VaR and ES",
        expect_intent="risk",
        expect_has_calc_id=True,
        expect_grounded=True,
        expect_keywords=["VaR", "ES"],
    ),
    EvalCase(
        name="gfc_stress",
        query="Run the 2008 GFC stress",
        expect_intent="stress",
        expect_has_calc_id=True,
        expect_grounded=True,
        expect_keywords=["P&L", "2008"],
    ),
    EvalCase(
        name="methodology_es",
        query="What is Expected Shortfall?",
        expect_intent="rag",
        expect_has_calc_id=False,
        expect_grounded=True,
        expect_keywords=["Shortfall", "VaR"],
    ),
    EvalCase(
        name="limits",
        query="Show limit utilization",
        expect_intent="limits",
        expect_has_calc_id=True,
        expect_grounded=True,
        expect_keywords=["Utilization", "Limit"],
    ),
    EvalCase(
        name="generative_ai_bubble",
        query="Generate an AI bubble unwind stress and show impact",
        expect_intent="stress",
        expect_has_calc_id=True,
        expect_grounded=True,
        expect_keywords=["P&L", "Stress"],
    ),
]


@dataclass
class CaseResult:
    name: str
    passed: bool
    details: Dict[str, Any]


def run_eval_suite(include_ragas: bool = True) -> Dict[str, Any]:
    """Run all cases against build_and_run and evaluate accuracy + RAGAS metrics."""
    from quantagent_desk.agents.graph import build_and_run, router_node
    from quantagent_desk.evals.ragas_eval import get_ragas_evaluator

    ragas = get_ragas_evaluator() if include_ragas else None
    results: List[CaseResult] = []
    ragas_scores: List[Dict[str, float]] = []
    passed = 0

    for case in EVAL_CASES:
        intent = router_node({"query": case.query}).get("intent", "")
        resp = build_and_run(case.query, session_id=f"eval-{case.name}", portfolio_id="FIRM")

        checks = {
            "intent_ok": intent == case.expect_intent,
            "calc_id_ok": (bool(resp.calculation_ids) == case.expect_has_calc_id)
            if case.expect_has_calc_id
            else True,
            "grounded_ok": resp.grounded == case.expect_grounded,
            "keywords_ok": all(
                any(kw.lower() in resp.answer.lower() for kw in [k])
                for k in case.expect_keywords
            )
            if case.expect_keywords
            else True,
        }

        ragas_res = None
        if ragas:
            ragas_res = ragas.evaluate_case(
                case_name=case.name,
                query=case.query,
                answer=resp.answer,
                context_docs=list(resp.research or []),
                calculation_ids=resp.calculation_ids,
                snapshot=resp.risk_snapshot.model_dump() if resp.risk_snapshot else None,
                stress_result=resp.stress_result.model_dump() if resp.stress_result else None,
            )
            ragas_scores.append({
                "faithfulness": ragas_res.faithfulness,
                "answer_relevancy": ragas_res.answer_relevancy,
                "context_precision": ragas_res.context_precision,
                "numeric_grounding": ragas_res.numeric_grounding,
                "overall_score": ragas_res.overall_score,
            })
            checks["ragas_passed"] = ragas_res.passed

        ok = all(checks.values())
        if ok:
            passed += 1
        results.append(
            CaseResult(
                name=case.name,
                passed=ok,
                details={
                    "intent": intent,
                    "calc_ids": resp.calculation_ids,
                    "grounded": resp.grounded,
                    "warnings": resp.warnings,
                    "checks": checks,
                    "ragas": {
                        "faithfulness": ragas_res.faithfulness,
                        "answer_relevancy": ragas_res.answer_relevancy,
                        "context_precision": ragas_res.context_precision,
                        "numeric_grounding": ragas_res.numeric_grounding,
                        "overall_score": ragas_res.overall_score,
                    } if ragas_res else None,
                    "answer_preview": resp.answer[:180],
                    "token_use": dict(resp.token_use or {}),
                },
            )
        )

    # Aggregate RAGAS summary
    ragas_summary = {}
    if ragas_scores:
        import numpy as np
        ragas_summary = {
            "mean_faithfulness": round(float(np.mean([s["faithfulness"] for s in ragas_scores])), 3),
            "mean_answer_relevancy": round(float(np.mean([s["answer_relevancy"] for s in ragas_scores])), 3),
            "mean_context_precision": round(float(np.mean([s["context_precision"] for s in ragas_scores])), 3),
            "mean_numeric_grounding": round(float(np.mean([s["numeric_grounding"] for s in ragas_scores])), 3),
            "mean_ragas_score": round(float(np.mean([s["overall_score"] for s in ragas_scores])), 3),
        }

    research_tokens = [
        int((c.details.get("token_use") or {}).get("research") or 0) for c in results
    ]
    prompt_tokens = [
        int((c.details.get("token_use") or {}).get("prompt") or 0) for c in results
    ]
    answer_tokens = [
        int((c.details.get("token_use") or {}).get("answer") or 0) for c in results
    ]
    gate_passed = (
        (passed / len(EVAL_CASES) if EVAL_CASES else 0.0) >= MIN_PASS_RATE
        and float(ragas_summary.get("mean_ragas_score", 0.0)) >= MIN_RAGAS_SCORE
        and float(ragas_summary.get("mean_numeric_grounding", 0.0)) >= MIN_NUMERIC_GROUNDING
    )

    return {
        "total": len(EVAL_CASES),
        "passed": passed,
        "failed": len(EVAL_CASES) - passed,
        "pass_rate": passed / len(EVAL_CASES) if EVAL_CASES else 0.0,
        "ragas_summary": ragas_summary,
        "token_use": {
            "mean_research_tokens": round(float(sum(research_tokens) / len(research_tokens)), 1) if research_tokens else 0.0,
            "mean_prompt_tokens": round(float(sum(prompt_tokens) / len(prompt_tokens)), 1) if prompt_tokens else 0.0,
            "mean_answer_tokens": round(float(sum(answer_tokens) / len(answer_tokens)), 1) if answer_tokens else 0.0,
        },
        "gate": {
            "passed": gate_passed,
            "min_pass_rate": MIN_PASS_RATE,
            "min_ragas_score": MIN_RAGAS_SCORE,
            "min_numeric_grounding": MIN_NUMERIC_GROUNDING,
        },
        "cases": [
            {"name": r.name, "passed": r.passed, **r.details} for r in results
        ],
    }


def print_report(summary: Optional[Dict[str, Any]] = None) -> None:
    summary = summary or run_eval_suite()
    print(f"\n=======================================================")
    print(f"  QuantAgent-Desk Evaluation Report & RAGAS Scorecard  ")
    print(f"=======================================================")
    print(f"Status: {summary['passed']}/{summary['total']} passed ({summary['pass_rate']:.0%})\n")

    rs = summary.get("ragas_summary", {})
    if rs:
        print("RAGAS Scorecard:")
        print(f"  • Faithfulness:         {rs.get('mean_faithfulness', 0):.1%}")
        print(f"  • Answer Relevancy:     {rs.get('mean_answer_relevancy', 0):.1%}")
        print(f"  • Context Precision:    {rs.get('mean_context_precision', 0):.1%}")
        print(f"  • Numeric Grounding:    {rs.get('mean_numeric_grounding', 0):.1%}")
        print(f"  • Overall RAGAS Score:  {rs.get('mean_ragas_score', 0):.1%}\n")

    tokens = summary.get("token_use") or {}
    if tokens:
        print(
            "Token use: "
            f"research {tokens.get('mean_research_tokens', 0):.0f} avg, "
            f"shock prompt {tokens.get('mean_prompt_tokens', 0):.0f} avg, "
            f"answer {tokens.get('mean_answer_tokens', 0):.0f} avg"
        )
    gate = summary.get("gate") or {}
    if gate:
        print(f"Gate: {'PASS' if gate.get('passed') else 'FAIL'}\n")

    for c in summary["cases"]:
        status = "PASS" if c["passed"] else "FAIL"
        r_info = ""
        if c.get("ragas"):
            r_info = f" | RAGAS={c['ragas']['overall_score']:.0%} (Grd={c['ragas']['numeric_grounding']:.0%})"
        print(f"  [{status}] {c['name']:<22} intent={c.get('intent'):<8} grounded={c.get('grounded')}{r_info}")
        if not c["passed"]:
            print(f"         checks={c.get('checks')}")

