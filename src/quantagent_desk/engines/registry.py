"""Calculation Registry for tracking, verifying, and auditing quantitative metrics."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional, Set


@dataclass
class RegisteredCalculation:
    calculation_id: str
    calc_type: str
    portfolio_id: str
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    known_numbers: Set[float] = field(default_factory=set)


class CalculationRegistry:
    """Thread-safe LRU registry for quantitative calculations."""

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._records: OrderedDict[str, RegisteredCalculation] = OrderedDict()
        self._lock = Lock()

    def register(
        self,
        calculation_id: str,
        calc_type: str,
        portfolio_id: str = "UNKNOWN",
        payload: Optional[Dict[str, Any]] = None,
    ) -> RegisteredCalculation:
        """Register a calculation output with extracted numbers for grounding audits."""
        payload = payload or {}
        numbers = self._extract_numbers_from_payload(payload)

        record = RegisteredCalculation(
            calculation_id=calculation_id,
            calc_type=calc_type,
            portfolio_id=portfolio_id,
            payload=payload,
            known_numbers=numbers,
        )

        with self._lock:
            if calculation_id in self._records:
                self._records.move_to_end(calculation_id)
            self._records[calculation_id] = record
            if len(self._records) > self._max_size:
                self._records.popitem(last=False)

        return record

    def get(self, calculation_id: str) -> Optional[RegisteredCalculation]:
        """Retrieve calculation record by ID."""
        with self._lock:
            record = self._records.get(calculation_id)
            if record:
                self._records.move_to_end(calculation_id)
            return record

    def all_ids(self) -> List[str]:
        """List all registered calculation IDs."""
        with self._lock:
            return list(self._records.keys())

    def clear(self) -> None:
        """Clear the registry (useful in tests)."""
        with self._lock:
            self._records.clear()

    @staticmethod
    def _extract_numbers_from_payload(payload: Any) -> Set[float]:
        """Extract all float/int numbers from a nested dict/list/model."""
        numbers: Set[float] = set()

        def recurse(val: Any) -> None:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numbers.add(float(val))
            elif isinstance(val, dict):
                for k, v in val.items():
                    # also extract numeric values from dict keys if applicable
                    recurse(v)
            elif isinstance(val, (list, tuple, set)):
                for item in val:
                    recurse(item)
            elif hasattr(val, "model_dump"):
                recurse(val.model_dump())
            elif hasattr(val, "__dict__"):
                recurse(val.__dict__)

        recurse(payload)
        return numbers


_REGISTRY: Optional[CalculationRegistry] = None


def get_registry() -> CalculationRegistry:
    """Return the singleton calculation registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = CalculationRegistry()
    return _REGISTRY
