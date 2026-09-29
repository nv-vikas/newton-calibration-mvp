"""Generate a visibly SYNTHETIC report without a GPU or customer data."""

import argparse
from pathlib import Path

from newton_calibration.reporting import build_report, load_recipe, render_report, write_bundle

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "src/newton_calibration/recipes/arm_joint_response.v1.json"


def make_demo(destination):
    recipe = load_recipe(RECIPE)
    facts = {
        "inputs": "Synthetic arm profile and two generated test records—not real robot measurements.",
        "discoveries": "Example controller and tool details are declared; physical setup has not been verified.",
        "limitations": "Illustrative values only. No Newton simulation or robot experiment was executed.",
        "scope": "Synthetic arm · reporting integration example",
        "next_action": "Connect a real run adapter. Keep missing setup facts visible before attempting calibration.",
        "execution_note": "This is a renderer fixture, not five executed calls. Never cite these values as performance evidence.",
        "baseline_error": 0.4,
        "tuned_error": 0.2,
        "metric_label": "synthetic held-out position RMSE",
        "metric_unit": "deg",
        "validation_passed": False,
        "activation_allowed": False,
        "parameters": {"example_only": ["stiffness", "damping"], "bounds": "Not hardware-qualified"},
        "objective": {"illustration": "Position error; not an executed objective"},
        "optimizer": {"name": "none", "evaluations": 0},
        "collection": {"generated_robot_motions": 0, "preview": "not_run"},
        "runtime": {"backend": "none"},
        "gates": {"physical_validation": False},
        "readiness": {"real_evidence": False, "operator_confirmation": False},
        "controller_tool": {"controller": "illustrative joint PD", "gripper": "not verified"},
        "split": {"train": ["synthetic-a"], "heldout": ["synthetic-b"]},
        "uncertainty": {"status": "not estimated"},
        "outputs": {"artifact": "customer HTML only"},
    }
    descriptions = {
        "analyze": "Inputs and missing confirmations are visible.",
        "plan": "Required settings and evidence are specified.",
        "fit": "Optimizer work would be recorded here; none was run.",
        "validate": "Show separate test results, limits and failed checks.",
        "write": "Preserve the record without granting activation.",
    }
    for stage, summary in descriptions.items():
        facts[f"{stage}.status"] = "not_run"
        facts[f"{stage}.summary"] = summary
    assert set(facts) == {field["id"] for field in recipe["reporting"]["fields"]}
    return write_bundle(
        RECIPE,
        destination,
        run_id="synthetic-report-example",
        execution_kind="synthetic",
        records={"synthetic": facts},
        facts={key: ("synthetic", "/" + key) for key in facts},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/report-demo")
    args = parser.parse_args()
    bundle = make_demo(args.output)
    print(render_report(build_report(RECIPE, bundle), Path(args.output) / "report.html"))
