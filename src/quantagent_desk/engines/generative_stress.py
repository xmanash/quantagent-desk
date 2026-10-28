"""Generative stress agent: LLM narrative → RiskFactorShock list.

The model sees a short instruction and a clipped narrative, and must reply
with JSON shocks only. If Ollama is down or the draft is unusable, the
template library fills in. Guardrails still cap the shocks that run.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import httpx

from quantagent_desk.guardrails.validator import get_guardrails
from quantagent_desk.models.core import RiskFactorShock, StressScenario
from quantagent_desk.utils.tokens import (
    NARRATIVE_TOKEN_BUDGET,
    SHOCK_COMPLETION_TOKEN_BUDGET,
    SHOCK_PROMPT_TOKEN_BUDGET,
    clip_text,
    estimate_tokens,
)

# pattern → (name, description, shocks). Fallback when the LLM is unavailable.
_TEMPLATES: List[Tuple[re.Pattern, str, str, List[RiskFactorShock]]] = [
    (
        re.compile(r"ai\s*bubble|tech\s*crash|nasdaq\s*crash|mag\s*7", re.I),
        "AI / Tech Bubble Unwind",
        "Sharp equity sell-off in AI/tech names, vol spike, modest credit widening.",
        [
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.25),
            RiskFactorShock(factor="VIX", shock_type="absolute", value=25.0),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.006),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=-0.005),
        ],
    ),
    (
        re.compile(r"stagflation|inflation\s*shock|cpi\s*spike", re.I),
        "Stagflation Shock",
        "Rates up sharply, equities down, commodities up, credit wider.",
        [
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=0.020),
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.15),
            RiskFactorShock(factor="WTI", shock_type="relative", value=0.30),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.010),
        ],
    ),
    (
        re.compile(r"china\s*hard\s*landing|em\s*crisis|emerging\s*market", re.I),
        "EM / China Hard Landing",
        "EM FX and equity stress, commodity drop, flight-to-quality in rates.",
        [
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.12),
            RiskFactorShock(factor="WTI", shock_type="relative", value=-0.25),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=-0.008),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.012),
        ],
    ),
    (
        re.compile(r"liquidity\s*crunch|funding\s*stress|repo\s*crisis", re.I),
        "Liquidity / Funding Crunch",
        "Vol spike, equity drop, credit blow-out, rates mixed.",
        [
            RiskFactorShock(factor="VIX", shock_type="absolute", value=35.0),
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.18),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.020),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=0.005),
        ],
    ),
    (
        re.compile(r"geopolitical|war\s*shock|oil\s*shock|middle\s*east", re.I),
        "Geopolitical / Oil Shock",
        "Oil spike, equity risk-off, rates down (flight to quality), credit wider.",
        [
            RiskFactorShock(factor="WTI", shock_type="relative", value=0.40),
            RiskFactorShock(factor="SPX", shock_type="relative", value=-0.10),
            RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=-0.010),
            RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.008),
        ],
    ),
]

ALLOWED_FACTORS = {"SPX", "VIX", "CDX_IG", "USD_10Y", "WTI", "EURUSD"}
_FACTOR_ALIASES = {
    "OIL": "WTI",
    "CRUDE": "WTI",
    "SP500": "SPX",
    "SPX500": "SPX",
    "US10Y": "USD_10Y",
    "UST10Y": "USD_10Y",
    "CREDIT": "CDX_IG",
    "CDX": "CDX_IG",
    "FX": "EURUSD",
    "EUR": "EURUSD",
}
_RATE_FACTORS = {"USD_10Y", "CDX_IG"}

# One tight instruction. The narrative is clipped separately.
SHOCK_SYSTEM_PROMPT = (
    "Map a risk narrative to at most 4 market shocks. "
    "Factors: SPX, VIX, CDX_IG, USD_10Y, WTI, EURUSD. "
    "SPX/WTI/EURUSD relative values are decimal returns (SPX -0.20 means -20%). "
    "USD_10Y and CDX_IG absolute values are decimals (0.01 = +100bp). "
    "VIX absolute is index points. "
    'Reply with JSON only: {"name":"","description":"","shocks":'
    '[{"factor":"SPX","shock_type":"relative","value":-0.20}]}'
)

_ollama_down = False


def _llm_disabled() -> bool:
    return os.getenv("QUANTAGENT_DISABLE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _parse_json_object(text: str) -> Optional[dict]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _ollama_draft(narrative: str) -> Tuple[Optional[dict], int, int]:
    """Call the local Ollama chat API. Returns (payload, prompt tokens, completion tokens)."""
    global _ollama_down
    if _ollama_down or _llm_disabled():
        return None, 0, 0

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": SHOCK_COMPLETION_TOKEN_BUDGET},
        "messages": [
            {"role": "system", "content": SHOCK_SYSTEM_PROMPT},
            {"role": "user", "content": narrative},
        ],
    }
    try:
        response = httpx.post(f"{host}/api/chat", json=body, timeout=2.5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        _ollama_down = True
        return None, 0, 0

    message = (payload.get("message") or {}).get("content", "")
    draft = _parse_json_object(message)
    prompt_tokens = int(payload.get("prompt_eval_count") or estimate_tokens(SHOCK_SYSTEM_PROMPT + narrative))
    completion_tokens = int(payload.get("eval_count") or estimate_tokens(message))
    return draft, prompt_tokens, completion_tokens


def _coerce_shocks(payload: dict) -> List[RiskFactorShock]:
    shocks: List[RiskFactorShock] = []
    for raw in payload.get("shocks") or []:
        if not isinstance(raw, dict):
            continue
        factor = str(raw.get("factor", "")).upper().replace(" ", "_")
        factor = _FACTOR_ALIASES.get(factor, factor)
        if factor not in ALLOWED_FACTORS:
            continue
        shock_type = raw.get("shock_type", "relative")
        if shock_type not in ("relative", "absolute"):
            shock_type = "absolute" if factor in _RATE_FACTORS or factor == "VIX" else "relative"
        try:
            value = float(raw.get("value"))
        except (TypeError, ValueError):
            continue
        # Models sometimes emit basis points (150) instead of 0.015.
        if factor in _RATE_FACTORS and abs(value) > 2:
            value = value / 10_000.0
        shocks.append(RiskFactorShock(factor=factor, shock_type=shock_type, value=value))
        if len(shocks) == 4:
            break
    return shocks


def _clamp_shocks(shocks: List[RiskFactorShock]) -> List[str]:
    """Pull shocks inside policy limits and report what moved."""
    warnings: List[str] = []
    policy = get_guardrails().policy.get("stress", {})
    max_eq = float(policy.get("max_equity_shock_abs", 0.60))
    max_rate = float(policy.get("max_rate_shock_bps", 500)) / 10_000.0
    for shock in shocks:
        if shock.factor == "SPX" and abs(shock.value) > max_eq:
            signed = -max_eq if shock.value < 0 else max_eq
            warnings.append(f"SPX shock clamped from {shock.value:.0%} to {signed:.0%}.")
            shock.value = signed
        if shock.factor in _RATE_FACTORS and shock.shock_type == "absolute" and abs(shock.value) > max_rate:
            signed = -max_rate if shock.value < 0 else max_rate
            warnings.append(
                f"{shock.factor} shock clamped from {shock.value * 10_000:.0f}bp to {signed * 10_000:.0f}bp."
            )
            shock.value = signed
    return warnings


@dataclass
class NarrativeBuild:
    scenario: Optional[StressScenario]
    warnings: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    source: str = "template"


class GenerativeStressAgent:
    """Turn a natural-language narrative into a StressScenario."""

    def __init__(self, llm: Optional[Callable[[str], Optional[dict]]] = None):
        self.guardrails = get_guardrails()
        self._llm = llm

    def generate(self, narrative: str) -> NarrativeBuild:
        """
        Returns a scenario plus token counts.
        scenario is None if policy blocks the shocks.
        """
        narrative = clip_text(narrative.strip(), NARRATIVE_TOKEN_BUDGET)
        warnings: List[str] = []
        shocks: List[RiskFactorShock] = []
        name = ""
        description = ""
        source = "template"
        prompt_tokens = 0
        completion_tokens = 0

        draft, prompt_tokens, completion_tokens = self._draft_from_llm(narrative)
        if draft:
            shocks = _coerce_shocks(draft)
            if shocks:
                source = "llm"
                name = clip_text(str(draft.get("name") or "Narrative stress"), 20)
                description = clip_text(str(draft.get("description") or narrative), 60)
            else:
                warnings.append("LLM shock draft had no usable factors; used the template library.")

        if not shocks:
            name, description, shocks = self._from_template(narrative)
            source = "template"
            prompt_tokens = 0
            completion_tokens = 0

        warnings.extend(_clamp_shocks(shocks))
        shock_dicts = [s.model_dump() for s in shocks]
        policy = self.guardrails.check_stress_shocks(shock_dicts)
        if not policy.allowed:
            return NarrativeBuild(
                scenario=None,
                warnings=[policy.message],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source=source,
            )
        warnings.extend(policy.warnings)

        if estimate_tokens(SHOCK_SYSTEM_PROMPT) > SHOCK_PROMPT_TOKEN_BUDGET:
            warnings.append("Shock instruction exceeds its token budget.")

        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        generated_by = f"ollama:{model}" if source == "llm" else "template_library"
        sources = ["LLM narrative → shocks"] if source == "llm" else ["Template library"]
        sources.append("Policy-capped shocks")

        scenario = StressScenario(
            name=name or "Generative Stress",
            description=description or narrative[:200],
            type="Generative",
            shocks=shocks,
            generated_by=generated_by,
            grounded_sources=sources,
            tags=["generative", "narrative", source],
        )
        return NarrativeBuild(
            scenario=scenario,
            warnings=warnings,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source=source,
        )

    def _draft_from_llm(self, narrative: str) -> Tuple[Optional[dict], int, int]:
        prompt_tokens = estimate_tokens(SHOCK_SYSTEM_PROMPT) + estimate_tokens(narrative)
        if self._llm is not None:
            payload = self._llm(narrative)
            completion = estimate_tokens(json.dumps(payload)) if payload else 0
            return payload, prompt_tokens, completion
        if _llm_disabled():
            return None, 0, 0
        return _ollama_draft(narrative)

    def _from_template(self, narrative: str) -> Tuple[str, str, List[RiskFactorShock]]:
        for pattern, name, desc, template_shocks in _TEMPLATES:
            if pattern.search(narrative):
                return name, desc, list(template_shocks)
        return (
            "Narrative Stress (default)",
            f"Generative scenario derived from: {narrative[:120]}",
            [
                RiskFactorShock(factor="SPX", shock_type="relative", value=-0.10),
                RiskFactorShock(factor="USD_10Y", shock_type="absolute", value=0.005),
                RiskFactorShock(factor="CDX_IG", shock_type="absolute", value=0.005),
                RiskFactorShock(factor="VIX", shock_type="absolute", value=15.0),
            ],
        )

    def list_templates(self) -> List[str]:
        return [name for _, name, _, _ in _TEMPLATES]
