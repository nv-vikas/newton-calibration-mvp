"""Read existing LOCAL exploratory artifacts. Never publishes them or reruns physics.

This adapter intentionally labels the historical runner as a legacy import. The
reporting contract is attached retrospectively; it did not govern the original fit.
"""

import argparse
import shutil
from pathlib import Path

from newton_calibration.core.io import atomic_write_json
from newton_calibration.reporting import build_report, render_report, write_bundle
from newton_calibration.reporting.report import ReportError, _json, digest

RECIPE = Path(__file__).resolve().parents[2] / "src/newton_calibration/recipes/arm_joint_response.v1.json"


def import_flexiv(original, replay, audit_path, destination, media_dir=None):
    original, replay, destination = Path(original), Path(replay), Path(destination)
    paths = {"audit": Path(audit_path)}
    for label, root, names in (
        ("original", original, ("plan", "runtime", "result", "candidates", "frozen_parameters")),
        ("replay", replay, ("plan", "runtime", "result", "frozen_parameters", "independent_verification")),
    ):
        paths.update({f"{label}_{name}": root / f"{name}.json" for name in names})
    raw = {key: path.read_bytes() for key, path in paths.items()}
    records = {key: _json(value) for key, value in raw.items()}
    plan, result, audit = records["original_plan"], records["replay_result"], records["audit"]
    replay_plan = records["replay_plan"]
    if result.get("activation_allowed") is not False or result.get("qualified_mvp1") is not False:
        raise ReportError("This historical importer only represents non-activatable exploratory results")
    if records["original_result"].get("status") != "exploratory_fit_completed":
        raise ReportError("Original exploratory fit is not recorded as completed")
    if result.get("optimizer_calls") != 0 or result.get("parameters_changed") is not False:
        raise ReportError("This adapter only supports the frozen-parameter replay")
    if raw["original_frozen_parameters"] != raw["replay_frozen_parameters"]:
        raise ReportError("Frozen parameters differ")
    for key, source in (
        ("prior_plan_sha256", "original_plan"),
        ("prior_result_sha256", "original_result"),
        ("prior_parameters_sha256", "original_frozen_parameters"),
    ):
        if replay_plan.get(key) != digest(raw[source]):
            raise ReportError(f"Replay source linkage failed: {key}")
    if plan["usd_sha256"] != replay_plan["usd_sha256"]:
        raise ReportError("Original and replay USD differ")
    if replay_plan.get("primary_variant") != "shim_start":
        raise ReportError("Primary replay variant changed; do not silently select a better result")
    if result["heldout"] != result["variants"]["heldout_shim_start"]:
        raise ReportError("Headline must use the declared primary held-out variant")
    captures = audit["captures"]
    if {plan["train"], plan["heldout"]} != {capture["name"] for capture in captures}:
        raise ReportError("Audit does not cover exactly the fit and held-out recordings")
    count = len(records["original_candidates"])
    joints = len(result["heldout"]["baseline_rmse_deg"])
    parameter_count = joints * len(plan["parameters_per_joint"])
    stage_status = "completed" if result["numeric_gate_passed"] is True else "failed"
    view = {
        "adapter": "flexiv-frozen-replay-report/v1",
        "source_sha256": {key: digest(value) for key, value in raw.items()},
        "inputs": f"Robot USD + {len(captures)} real joint recordings; {sum(c['samples'] for c in captures):,} feedback samples.",
        "discoveries": f"{result['heldout']['changed_command_frames']} held-out command targets changed after reconstructing the ROS bridge limiter.",
        "limitations": "Joint mapping, applied gains and tool dynamics remain unconfirmed. Effective response settings—not uniquely identified hardware constants.",
        "next_action": "Confirm mapping, controller settings and tool dynamics; broaden motion evidence before qualifying the package.",
        "execution_note": "A separate exploratory runner performed the fit after the guarded workflow blocked on unconfirmed mapping. These are stages, not five completed public API calls. The later replay reused frozen parameters.",
        "scope": f"Flexiv arm · {joints} joints · free-space response · empty attached Grav gripper",
        "metric_label": "held-out joint-position RMSE over the full replay time grid",
        "metric_unit": "deg",
        "parameters": {
            "bounds": plan["parameters_per_joint"],
            "selected": result["parameters"],
            "ownership": {
                "stiffness_scale": "Newton drive",
                "damping_scale": "Newton drive",
                "input_filter_tau_s": "runner command filter",
            },
        },
        "objective": {
            "definition": plan["objective"],
            "metric_not_used": "No force, contact or policy-transfer objective",
        },
        "optimizer": {
            "original_method": plan["optimizer"],
            "evaluated_configurations": count,
            "population": plan["population"],
            "generations": plan["generations"],
            "replay_optimizer_calls": result["optimizer_calls"],
        },
        "collection": {
            "reused": [plan["train"], plan["heldout"]],
            "new_motions_generated": False,
            "robot_commands_sent": result["real_robot_commands_sent"],
            "collection_preview_generated": False,
            "next_evidence": "Published targets, device timestamps, broader independent motion families",
        },
        "gates": {
            "thresholds": plan["validation_gates"],
            "numeric_gate_passed": result["numeric_gate_passed"],
            "independent_verification": records["replay_independent_verification"]["all_checks_passed"],
        },
        "readiness": {
            "captures": [
                {
                    key: c[key]
                    for key in (
                        "name",
                        "samples",
                        "duration_s",
                        "median_rate_hz",
                        "finite_array_checks_passed",
                        "gaps_over_1p5_nominal_intervals",
                    )
                }
                for c in captures
            ],
            "mapping": replay_plan["mapping_assumption"],
        },
        "controller_tool": {
            "runtime_controller": replay_plan["fixed_runtime"]["controller"],
            "real_command_path": replay_plan["command_assumption"],
            "tool": replay_plan["tool"],
        },
        "split": {
            "train": plan["train"],
            "heldout": plan["heldout"],
            "disclosure": "Both inspected for quality; held-out capture excluded from parameter fitting.",
        },
        "uncertainty": {
            "identification": "Effective response only; no parameter confidence intervals established.",
            "timing_sensitivity": "send_end replay is a sensitivity check, not an uncertainty bound or selected model.",
        },
        "outputs": {
            "frozen_parameters": records["replay_frozen_parameters"],
            "qualified_mvp1": result["qualified_mvp1"],
            "activation_allowed": result["activation_allowed"],
            "interpretation": result["interpretation"],
        },
        "analyze.status": "blocked",
        "analyze.summary": "Real recordings inspected; setup confirmations remained open.",
        "plan.status": "recorded",
        "plan.summary": f"{parameter_count} response settings across {joints} joints; runtime fixed.",
        "fit.status": "completed",
        "fit.summary": f"{count} candidate configurations tested in the original search. Zero new optimizer calls in replay.",
        "validate.status": stage_status,
        "validate.summary": "Frozen settings compared on the separate held-out recording. Numerical check is not transfer proof.",
        "write.status": "blocked",
        "write.summary": "Experimental results preserved; no approved loadable calibration package.",
    }
    facts = {key: ("view", "/" + key) for key in view if key not in {"adapter", "source_sha256"}}
    facts.update(
        {
            "baseline_error": ("replay_result", "/heldout/overall_baseline_rmse_deg"),
            "tuned_error": ("replay_result", "/heldout/overall_tuned_rmse_deg"),
            "validation_passed": ("replay_result", "/numeric_gate_passed"),
            "activation_allowed": ("replay_result", "/activation_allowed"),
            "runtime": ("replay_plan", "/fixed_runtime"),
        }
    )
    records["view"] = view
    bundle_path = write_bundle(
        RECIPE,
        destination,
        run_id="flexiv-fit03-plus-frozen-replay01",
        execution_kind="legacy_import",
        records=records,
        facts=facts,
    )
    if media_dir:
        bundle = _json(bundle_path.read_bytes())
        bundle["media"] = []
        (destination / "media").mkdir()
        for joint in (1, 4):
            name = f"visual-joint-{joint}.gif"
            source = Path(media_dir) / name
            shutil.copy2(source, destination / "media" / name)
            bundle["media"].append(
                {
                    "path": f"media/{name}",
                    "sha256": digest(source.read_bytes()),
                    "mime": "image/gif",
                    "kind": "measured_simulation_trace",
                    "caption": f"Joint {joint}: measured (blue), untuned Newton (orange), tuned Newton (green). Error has its own scale",
                }
            )
        atomic_write_json(bundle_path, bundle)
    render_report(build_report(RECIPE, bundle_path), destination / "report.html")
    return bundle_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--media")
    args = parser.parse_args()
    print(import_flexiv(args.original, args.replay, args.audit, args.output, args.media))
