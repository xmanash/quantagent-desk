"""Full accuracy + RAGAS gate. CI runs this via scripts/run_evals.py."""

import os

import pytest

os.environ.setdefault("QUANTAGENT_DISABLE_LLM", "1")
os.environ.setdefault("QUANTAGENT_RAGAS_LEXICAL", "1")
os.environ.setdefault("QUANTAGENT_SKIP_EMBEDDINGS", "1")


@pytest.mark.eval
def test_accuracy_and_ragas_gate():
    from quantagent_desk.evals.suite import run_eval_suite

    summary = run_eval_suite()
    assert summary["failed"] == 0, summary["cases"]
    assert summary["gate"]["passed"] is True
    assert summary["ragas_summary"]["mean_ragas_score"] >= 0.70
    assert summary["ragas_summary"]["mean_numeric_grounding"] >= 0.80
    by_name = {case["name"]: case for case in summary["cases"]}
    assert by_name["firm_var"]["token_use"].get("research", 0) == 0
    assert by_name["methodology_es"]["token_use"].get("research", 0) > 0
