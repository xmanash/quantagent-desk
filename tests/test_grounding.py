"""Unverified money is removed without a model call."""

from quantagent_desk.guardrails.validator import Guardrails


def test_matching_var_is_kept():
    answer, warnings, grounded = Guardrails().ground_response(
        answer="VaR 95%: **$1,000**",
        calculation_ids=["calc-1"],
        citations=[],
        intent="risk",
        snapshot={"var_95": 1000, "limit_utilization": {"VaR_95": 0.25}},
    )
    assert "$1,000" in answer
    assert "95%" in answer
    assert grounded is True
    assert warnings == []


def test_unverified_dollars_are_redacted():
    answer, warnings, grounded = Guardrails().ground_response(
        answer="VaR is $9,999,999.",
        calculation_ids=["missing"],
        citations=[],
        intent="risk",
        snapshot={"var_95": 1000, "limit_utilization": {}},
    )
    assert "$9,999,999" not in answer
    assert "$[redacted]" in answer
    assert grounded is False
    assert warnings


def test_quoted_research_dollars_stay():
    answer, warnings, grounded = Guardrails().ground_response(
        answer="The worked example uses $2,500.",
        calculation_ids=[],
        citations=["es.md"],
        intent="rag",
        research=["Expected Shortfall example: $2,500."],
    )
    assert "$2,500" in answer
    assert grounded is True
    assert not any("Unverified" in w for w in warnings)


def test_bad_utilization_percent_is_removed():
    answer, _, grounded = Guardrails().ground_response(
        answer="ES 97.5%: **$1,000**\nVaR_95 12.3%",
        calculation_ids=["calc-1"],
        citations=[],
        intent="limits",
        snapshot={"var_95": 1000, "es_97_5": 1000, "limit_utilization": {"VaR_95": 0.40}},
    )
    assert "97.5%" in answer
    assert "12.3%" not in answer
    assert "[redacted]%" in answer
    assert "$1,000" in answer
    assert grounded is False
