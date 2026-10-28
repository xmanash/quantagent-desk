"""Guardrails: scope checks, numeric grounding, policy enforcement."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

POLICY_PATH = Path(__file__).with_name("policies.yaml")


def _load_policy() -> Dict[str, Any]:
    if POLICY_PATH.exists():
        with open(POLICY_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


from dataclasses import dataclass, field
from quantagent_desk.engines.registry import get_registry


class GuardrailResult:
    def __init__(self, allowed: bool, message: str = "", warnings: Optional[List[str]] = None):
        self.allowed = allowed
        self.message = message
        self.warnings = warnings or []


@dataclass
class NumericGroundingAudit:
    """Detailed audit for numeric grounding verification."""
    grounded: bool = True
    total_numbers_found: int = 0
    verified_numbers: List[float] = field(default_factory=list)
    unverified_numbers: List[float] = field(default_factory=list)
    hallucination_score: float = 0.0
    calculation_ids_verified: List[str] = field(default_factory=list)
    calculation_ids_missing: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class Guardrails:
    """Input/output validators + numeric grounding critic."""

    MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")
    BLOCKED_RE = re.compile(
        r"\b(place\s+(?:an?\s+)?order|execute\s+(?:a\s+)?trade|wire\s+transfer|password|api[_\s-]?key)\b",
        re.I,
    )

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = policy or _load_policy()

    def check_input(self, query: str) -> GuardrailResult:
        """Refuse out-of-scope or dangerous requests."""
        if self.BLOCKED_RE.search(query):
            msg = self.policy.get("response", {}).get(
                "off_scope_message",
                "Request outside risk desk scope.",
            )
            return GuardrailResult(False, msg)
        return GuardrailResult(True)

    def check_stress_shocks(self, shocks: List[Dict[str, Any]]) -> GuardrailResult:
        """Cap extreme generative shocks per policy."""
        stress_pol = self.policy.get("stress", {})
        max_eq = float(stress_pol.get("max_equity_shock_abs", 0.60))
        max_bps = float(stress_pol.get("max_rate_shock_bps", 500)) / 10000.0
        warnings: List[str] = []
        for s in shocks:
            factor = str(s.get("factor", "")).upper()
            val = abs(float(s.get("value", 0)))
            stype = s.get("shock_type", "relative")
            if "SPX" in factor or "EQUITY" in factor:
                if val > max_eq:
                    return GuardrailResult(
                        False,
                        f"Equity shock {val:.0%} exceeds policy max {max_eq:.0%}.",
                    )
            if "USD" in factor or "RATE" in factor or "10Y" in factor:
                if stype == "absolute" and val > max_bps:
                    return GuardrailResult(
                        False,
                        f"Rate shock {val*10000:.0f}bps exceeds policy max {max_bps*10000:.0f}bps.",
                    )
        return GuardrailResult(True, warnings=warnings)

    @classmethod
    def extract_monetary_values(cls, text: str) -> List[float]:
        """Extract all dollar numbers as clean floats."""
        raw_matches = re.findall(r"\$\s*(-?[\d,]+(?:\.\d+)?)", text)
        values: List[float] = []
        for raw in raw_matches:
            cleaned = raw.replace(",", "").strip()
            try:
                values.append(float(cleaned))
            except ValueError:
                pass
        return values

    def verify_numeric_grounding(
        self,
        answer: str,
        calculation_ids: Optional[List[str]] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        stress_result: Optional[Dict[str, Any]] = None,
        intent: str = "general",
        research: Optional[List[str]] = None,
    ) -> NumericGroundingAudit:
        """
        Verify every monetary figure in the answer against calculation payloads.
        Checks CalculationRegistry and explicit snapshot/stress data within tolerance.
        """
        calculation_ids = calculation_ids or []
        ng = self.policy.get("numeric_grounding", {})
        tol = float(ng.get("tolerance_pct", 0.005))
        registry = get_registry()

        audit = NumericGroundingAudit()
        known_numbers: set[float] = set()

        # 1. Resolve registered calculation numbers
        for cid in calculation_ids:
            rec = registry.get(cid)
            if rec:
                audit.calculation_ids_verified.append(cid)
                known_numbers.update(rec.known_numbers)
            else:
                audit.calculation_ids_missing.append(cid)

        # 2. Add numbers from passed snapshot / stress payloads and quoted research.
        if snapshot:
            known_numbers.update(registry._extract_numbers_from_payload(snapshot))
        if stress_result:
            known_numbers.update(registry._extract_numbers_from_payload(stress_result))
        for excerpt in research or []:
            known_numbers.update(self.extract_monetary_values(excerpt))

        # 3. Extract numbers from text
        monetary_vals = self.extract_monetary_values(answer)
        audit.total_numbers_found = len(monetary_vals)

        if not monetary_vals:
            audit.grounded = True
            return audit

        # 4. Check whether calculations are present when required
        if intent in ("risk", "stress", "limits") and ng.get("require_calculation_id_for_money", True):
            if not calculation_ids and not known_numbers:
                audit.grounded = False
                audit.unverified_numbers = monetary_vals
                audit.hallucination_score = 1.0
                audit.warnings.append("Ungrounded monetary figures blocked: no calculation ID provided.")
                return audit

        # 5. Cross-reference each extracted number against ground-truth values
        for val in monetary_vals:
            matched = False
            for known in known_numbers:
                diff = abs(val - known)
                threshold = max(1.0, abs(known) * tol)
                if diff <= threshold:
                    matched = True
                    break
            if matched:
                audit.verified_numbers.append(val)
            else:
                audit.unverified_numbers.append(val)

        if audit.total_numbers_found > 0:
            audit.hallucination_score = len(audit.unverified_numbers) / audit.total_numbers_found

        # 6. Apply grounding policy decisions
        if audit.unverified_numbers:
            unv_str = ", ".join(f"${n:,.0f}" for n in audit.unverified_numbers[:5])
            audit.warnings.append(f"Unverified monetary figures detected ({unv_str}) not matched in calculation ID.")
            audit.grounded = False
        else:
            audit.grounded = True

        return audit

    _PCT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
    _CONFIDENCE_PCTS = {95.0, 97.5, 99.0}

    def _redact_unverified_money(self, answer: str, unverified: List[float]) -> str:
        def replacer(match: re.Match) -> str:
            values = self.extract_monetary_values(match.group(0))
            if not values:
                return match.group(0)
            value = values[0]
            for bad in unverified:
                if abs(value - bad) <= 1e-6:
                    return "$[redacted]"
            return match.group(0)

        return re.sub(r"\$\s*-?[\d,]+(?:\.\d+)?", replacer, answer)

    def _known_utilization_pcts(self, snapshot: Optional[Dict[str, Any]]) -> List[float]:
        util = (snapshot or {}).get("limit_utilization") or {}
        known: List[float] = []
        for raw in util.values():
            try:
                known.append(float(raw) * 100.0)
            except (TypeError, ValueError):
                continue
        return known

    def _redact_unverified_percents(
        self,
        answer: str,
        snapshot: Optional[Dict[str, Any]],
        intent: str,
    ) -> Tuple[str, List[str]]:
        """Utilization percents must match the snapshot. Confidence levels stay."""
        if intent not in ("risk", "limits") or not snapshot:
            return answer, []
        known = self._known_utilization_pcts(snapshot)
        if intent == "limits" and not known:
            return answer, []
        warnings: List[str] = []

        def replacer(match: re.Match) -> str:
            shown = float(match.group(1))
            if any(abs(shown - level) < 0.05 for level in self._CONFIDENCE_PCTS):
                return match.group(0)
            if any(abs(shown - level) <= 0.6 for level in known):
                return match.group(0)
            warnings.append(f"Unverified utilization {shown:g}% removed.")
            return "[redacted]%"

        return self._PCT_RE.sub(replacer, answer), warnings

    def ground_response(
        self,
        answer: str,
        calculation_ids: List[str],
        citations: List[str],
        intent: str,
        snapshot: Optional[Dict[str, Any]] = None,
        stress_result: Optional[Dict[str, Any]] = None,
        research: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], bool]:
        """
        Check every dollar figure against a calculation, snapshot, or quoted excerpt.
        Unverified figures are removed. No model call.
        """
        warnings: List[str] = []
        ng = self.policy.get("numeric_grounding", {})
        strict = bool(ng.get("strict_rejection", False))
        redact = bool(ng.get("redact_unverified", True))
        max_unverified = int(ng.get("max_unverified_numbers", 0))

        audit = self.verify_numeric_grounding(
            answer=answer,
            calculation_ids=calculation_ids,
            snapshot=snapshot,
            stress_result=stress_result,
            intent=intent,
            research=research,
        )
        warnings.extend(audit.warnings)

        over_cap = len(audit.unverified_numbers) > max_unverified
        if over_cap and strict:
            refuse = self.policy.get("response", {}).get(
                "refuse_message",
                "Cannot return ungrounded numbers.",
            )
            return refuse, warnings, False

        if over_cap and redact:
            answer = self._redact_unverified_money(answer, audit.unverified_numbers)
            answer = answer.rstrip() + "\n\nUngrounded figure removed."

        answer, percent_warnings = self._redact_unverified_percents(answer, snapshot, intent)
        warnings.extend(percent_warnings)

        if intent == "rag" and ng.get("require_citation_for_methodology", True):
            if not citations and len(answer) > 80:
                warnings.append("Methodology answer lacks document citation")

        grounded = not over_cap and not percent_warnings
        return answer, warnings, grounded


def get_guardrails() -> Guardrails:
    return Guardrails()

