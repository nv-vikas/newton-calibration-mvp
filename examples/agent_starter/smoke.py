"""Synthetic API-host exercise, not an LLM evaluation or Newton/robot result."""

import argparse
import json
import runpy
from pathlib import Path

from newton_calibration import guided
from newton_calibration.agent_tools import CalibrationAgentTools
from newton_calibration.core.io import write_json


def smoke(destination):
    root = Path(destination).expanduser().resolve()
    # Reuse the explicitly synthetic example generator, not a second physics model.
    fixture = runpy.run_path(str(Path(__file__).parents[1] / "guided" / "demo.py"))["create_fixture"]
    asset, answers, evidence = fixture(root / "inputs")
    answers["evidence"] = evidence
    answers.pop("fit_budget")
    confirmations = answers.pop("confirm")
    setup_file = root / "inputs" / "synthetic_setup.json"
    write_json(setup_file, answers)
    opts = {
        "jobs_root": root / "jobs",
        "input_roots": [root / "inputs"],
        "generations": 2,
        "population": 4,
        "allowed_adapters": ("analytic",),
        "identity": "Synthetic host test",
    }
    tools = CalibrationAgentTools(**opts)
    created = tools.call(
        "start",
        {
            "job": "arm",
            "asset": str(asset),
            "goal": "Synthetic arm-response test",
            "recipe": "arm_joint_response@1",
            "reason": "Explicit test scope, not hardware",
        },
    )
    tools.call("inspect_source", {"path": str(setup_file)})
    tools.call(
        "submit",
        {
            "job": "arm",
            "expected_revision": created["revision"],
            "answers": answers,
            "sources": [str(setup_file)],
            "rationale": "Synthetic setup and CSVs",
        },
    )
    blocked = tools.call("run", {"job": "arm"})
    assert "fit" not in blocked["session"]["artifacts"]
    # Simulate the separate human-review channel. Never expose this as a model tool.
    guided.provide(
        root / "jobs" / "arm",
        {"confirm": confirmations},
        source="Synthetic review fixture only",
        confirmed_by="Synthetic test operator, not a real person",
    )
    prepared = tools.call("run", {"job": "arm"})
    approval_basis = (prepared["session_id"], prepared["revision"], prepared["session"]["artifacts"]["plan"]["sha256"])

    def test_host_approval(state):
        return (state["session_id"], state["revision"], state["artifacts"]["plan"]["sha256"]) == approval_basis

    # A fresh process/agent can use the same persisted job; no chat context needed.
    resumed = CalibrationAgentTools(**opts, authorize=test_host_approval)
    finished = resumed.call("run", {"job": "arm"})
    again = CalibrationAgentTools(**opts).call("run", {"job": "arm"})
    assert finished["recorded_state"] == again["recorded_state"] == "completed"
    assert finished["session"]["artifacts"] == again["session"]["artifacts"]
    assert not finished["activation_allowed"] and not finished["real_transfer_tested"]
    report = {
        "classification": "synthetic_analytic_contract_test",
        "newton_executed": False,
        "real_data_used": False,
        "llm_evaluated": False,
        "state": finished["recorded_state"],
        "scientific_calls": finished["scientific_calls"],
        "activation_allowed": False,
        "resume_preserved_artifacts": True,
        "customer_report": finished["customer_report"],
    }
    write_json(root / "smoke_result.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    smoke(parser.parse_args().output)
