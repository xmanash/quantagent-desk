"""Narrative → shock builder: LLM draft, template fallback, token cap."""

from quantagent_desk.engines.generative_stress import (
    SHOCK_SYSTEM_PROMPT,
    GenerativeStressAgent,
)
from quantagent_desk.utils.tokens import NARRATIVE_TOKEN_BUDGET, estimate_tokens


def test_system_prompt_stays_inside_budget():
    assert estimate_tokens(SHOCK_SYSTEM_PROMPT) <= 120


def test_llm_narrative_becomes_shocks():
    seen = {}

    def fake_llm(narrative: str) -> dict:
        seen["narrative"] = narrative
        return {
            "name": "Dollar squeeze",
            "description": "EURUSD drops and equities sell off.",
            "shocks": [
                {"factor": "EURUSD", "shock_type": "relative", "value": -0.08},
                {"factor": "SPX", "shock_type": "relative", "value": -0.12},
                {"factor": "oil", "shock_type": "relative", "value": 0.15},
            ],
        }

    build = GenerativeStressAgent(llm=fake_llm).generate("A sudden dollar squeeze hits EUR and oil.")
    assert build.source == "llm"
    assert build.scenario is not None
    assert build.scenario.generated_by.startswith("ollama:")
    factors = {shock.factor for shock in build.scenario.shocks}
    assert factors == {"EURUSD", "SPX", "WTI"}
    assert build.prompt_tokens > 0
    assert build.completion_tokens > 0
    assert estimate_tokens(seen["narrative"]) <= NARRATIVE_TOKEN_BUDGET


def test_llm_rate_bps_and_equity_clamp():
    def fake_llm(_narrative: str) -> dict:
        return {
            "name": "Hiking panic",
            "description": "Rates gap and equities crash.",
            "shocks": [
                {"factor": "USD_10Y", "shock_type": "absolute", "value": 250},
                {"factor": "SPX", "shock_type": "relative", "value": -0.95},
            ],
        }

    build = GenerativeStressAgent(llm=fake_llm).generate("Rates explode and stocks collapse.")
    assert build.scenario is not None
    by_factor = {shock.factor: shock for shock in build.scenario.shocks}
    assert abs(by_factor["USD_10Y"].value - 0.025) < 1e-9
    assert by_factor["SPX"].value == -0.60
    assert any("clamped" in warning.lower() for warning in build.warnings)


def test_empty_llm_draft_uses_template():
    build = GenerativeStressAgent(llm=lambda _n: {"shocks": []}).generate(
        "Generate an AI bubble unwind"
    )
    assert build.source == "template"
    assert build.scenario is not None
    assert build.scenario.name == "AI / Tech Bubble Unwind"
    assert build.prompt_tokens == 0
    assert any("template" in warning.lower() for warning in build.warnings)


def test_long_narrative_is_clipped_before_the_model():
    def fake_llm(narrative: str) -> dict:
        return {"name": "Clip", "description": narrative, "shocks": [
            {"factor": "SPX", "shock_type": "relative", "value": -0.05},
        ]}

    narrative = "stagflation " + ("inflation shock " * 400)
    build = GenerativeStressAgent(llm=fake_llm).generate(narrative)
    assert build.scenario is not None
    assert estimate_tokens(build.scenario.description) <= 80
    assert "…" in build.scenario.description or estimate_tokens(narrative) > NARRATIVE_TOKEN_BUDGET
