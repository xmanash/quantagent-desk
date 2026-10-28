#!/usr/bin/env python3
"""Run the accuracy / grounding evaluation suite (CI-friendly)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantagent_desk.evals.suite import print_report, run_eval_suite


def main():
    summary = run_eval_suite()
    print_report(summary)
    out = Path(__file__).resolve().parents[1] / "docs" / "eval_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out}")
    # Non-zero exit if accuracy or the RAGAS gate fails (for CI)
    gate_ok = (summary.get("gate") or {}).get("passed", summary["failed"] == 0)
    sys.exit(0 if gate_ok else 1)


if __name__ == "__main__":
    main()
