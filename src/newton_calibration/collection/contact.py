"""Data-only contact collection specifications. No driver, simulator or fitter.

Templates and pilot counts are collection proposals, never measurements,
identifiability results, executable trajectories or hardware authorization.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
from copy import deepcopy
from pathlib import Path

from newton_calibration.core.io import atomic_write_text, sha256_file, utc_now, write_json

SCHEMA = "newton.calibration.collection-recipe/v1"
MODES = {
    "mode": "collection_only",
    "fit_supported": False,
    "robot_execution_supported": False,
    "numeric_motion_generation_supported": False,
    "preview_supported": False,
}
SIGNALS = {
    "commands": {
        "file": "commands.jsonl",
        "record": "Final dispatched arm targets, semantics/units/frame, sequence, host send time; preserve requested targets separately",
        "rule": "Log downstream of limiter/interpolation where possible. Requested commands are not published commands.",
    },
    "robot_feedback": {
        "file": "feedback.jsonl",
        "record": "Native robot timestamp plus host receipt time; named q/dq and TCP/flange pose, TCP velocity and joint torque where available",
        "rule": "Record units, joint ordering, pose frame/quaternion order, source freshness and actual cadence.",
    },
    "gripper": {
        "file": "gripper_commands.jsonl + gripper_feedback.jsonl",
        "record": "Actual width/squeeze commands with units and native measured jaw opening; current/effort/status if available",
        "rule": "A setpoint or motor current is not calibrated normal force. Missing feedback stays missing.",
    },
    "video": {
        "file": "original camera files + frame timestamp sidecar",
        "record": "Fixed view of peg/fingers and peg/hole; visible trial ID and synchronization event; camera calibration/scale for metric claims",
        "rule": "Keep original frames/timestamps. Measure frame cadence and sync uncertainty; video is not a force sensor.",
    },
    "wrist_wrench": {
        "file": "wrist_wrench.jsonl or explicitly mapped native log",
        "record": "Fx/Fy/Fz/Mx/My/Mz in N and Nm; separate raw sensor and processed external wrench when present",
        "rule": "Verify installed sensor, SDK field names, frame AND moment origin, signs, tare, filtering, payload compensation and native times; zero-valued fields may mean no sensor.",
    },
    "relative_pose_depth": {
        "file": "relative_pose_depth.jsonl + calibration record",
        "record": "Measured peg relative to hole/fingers, insertion depth, uncertainty and source timestamps",
        "rule": "TCP travel is not peg insertion depth without verified grasp and frame transforms. Use suitable metrology for the required clearance.",
    },
    "jaw_normal_force": {
        "file": "normal_force.jsonl + calibration record or validated force map",
        "record": "Independent normal load with units, timing, calibration and per-finger/total convention",
        "rule": "Opposing jaw loads can cancel at the wrist. A wrist wrench is not a squeeze-force measurement.",
    },
    "tangential_force": {
        "file": "tangential_force.jsonl or known-load record",
        "record": "Independent tangential load/force, direction, calibration, timing and any inertial correction",
        "rule": "Friction identification needs independently known normal and tangential load; contact geometry/rolling must be controlled.",
    },
    "local_deflection": {
        "file": "local_deflection.jsonl + measurement calibration",
        "record": "Local normal displacement across the stated contact/assembly, units, source times and measurement uncertainty",
        "rule": "Robot TCP displacement includes arm/controller/fixture compliance; dynamic damping needs sufficient bandwidth.",
    },
}

SETUP_TEMPLATE = {
    "geometry": {
        "object_id": None,
        "object_material_finish": None,
        "object_mass_kg": None,
        "measured_dimensions_and_uncertainty": None,
        "finger_geometry_material": None,
        "peg_in_gripper_transform": None,
        "hole_id_and_as_built_dimensions": None,
        "chamfer_depth_fixture": None,
        "frame_transforms_and_units": None,
        "measurement_source": None,
    },
    "instrumentation": {
        "signal_mapping_file": None,
        "clock_mapping_and_uncertainty": None,
        "native_sample_rates_and_filtering": None,
        "camera_views_and_calibration": None,
        "wrist_sensor_identity": None,
        "wrench_frame_origin_tare_compensation": None,
        "jaw_force_reference_and_convention": None,
        "capabilities": {key: "unknown" for key in SIGNALS},
    },
    "trial_settings": {
        "condition_values_and_units": None,
        "actual_deployment_mode": None,
        "approved_start_pose": None,
        "approved_speed_acceleration": None,
        "approved_force_torque_depth_limits": None,
        "grasp_squeeze_levels": None,
        "insertion_offsets_tilts_rates": None,
        "success_jam_timeout_abort_criteria": None,
        "heldout_conditions_and_seed": None,
    },
    "operator_review": {
        "responsible_operator": None,
        "approved_protocol_reference": None,
        "stop_and_safe_recovery_procedure": None,
        "fixture_and_catch_review": None,
    },
    "upstream": {
        "mvp1_package_path_and_sha256": None,
        "mvp2_package_path_and_sha256": None,
        "current_robot_gripper_configuration_snapshot": None,
    },
}
SETUP_QUESTIONS = {
    "geometry": "Measure the actual peg, fingers and (for insertion) selected hole, mass, material/finish and frame transforms.",
    "instrumentation": "Confirm which signals can be logged, their units/frames, force source and clock alignment.",
    "trial_settings": "Have the robot engineer specify the deployment controller, numeric conditions, stop limits and held-out cases.",
    "operator_review": "Identify the responsible operator and the reviewed protocol, fixture/catch and stop/recovery procedure.",
    "upstream": "Record existing MVP1/MVP2 package references if available, or preserve the current configuration snapshot. Missing packages do not prevent planning data collection.",
}


def load_collection_recipe(path):
    raw = Path(path).read_bytes()
    definition = json.loads(raw)
    json.dumps(definition, allow_nan=False)
    if definition.get("schema") != SCHEMA or definition.get("execution") != MODES:
        raise ValueError("Contact recipes must be explicitly collection-only; no execution capability may be enabled")
    for key in ("id", "title", "purpose", "stage"):
        if not isinstance(definition.get(key), str) or not definition[key].strip():
            raise ValueError(f"Collection recipe requires {key}")
    if definition["stage"] not in {"mvp2", "mvp3"} or definition.get("guided", {}).get("executor") is not None:
        raise ValueError("Unsupported collection stage or executor")
    for key in ("minimum_kit", "parameters", "trials", "metrics_to_record", "limitations", "conditional_experiments"):
        if not isinstance(definition.get(key), list) or not definition[key]:
            raise ValueError(f"Collection recipe requires nonempty {key}")
    codes = set()
    for trial in definition["trials"]:
        code = trial.get("code")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", code) or code in codes:
            raise ValueError("Invalid/duplicate trial code")
        codes.add(code)
        if trial.get("split") not in {"development", "heldout"} or trial.get("priority") not in {"core", "conditional"}:
            raise ValueError("Every trial needs an explicit split and priority")
        if type(trial.get("repeats")) is not int or not 1 <= trial["repeats"] <= 100:
            raise ValueError("Trial repeat count must be an integer between 1 and 100")
        if not trial.get("requires") or set(trial["requires"]) - SIGNALS.keys():
            raise ValueError("Trial references unknown evidence signals")
        for key in ("motion", "condition"):
            if not isinstance(trial.get(key), str) or not trial[key].strip():
                raise ValueError(f"Trial requires {key}")
    if not any(t["split"] == "heldout" for t in definition["trials"]):
        raise ValueError("Collection needs independent held-out trials")
    reporting = definition.get("reporting", {})
    if not reporting.get("required_sections") or not reporting.get("result_source"):
        raise ValueError("Collection recipe must define required reporting and record provenance")
    definition["_sha256"] = hashlib.sha256(raw).hexdigest()
    return definition


def inspect_contact_inputs(session, definition):
    """Inspect declarations only. Never call five-call analyze or parse recordings."""
    inputs = session["inputs"]
    setup = inputs.get("contact_setup") or {}
    if not isinstance(setup, dict) or set(setup) - SETUP_TEMPLATE.keys():
        raise ValueError(
            "contact_setup must contain only geometry, instrumentation, trial_settings, operator_review, upstream"
        )
    for key, value in setup.items():
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"contact_setup.{key} must be an object or null")
        if value and set(value) - SETUP_TEMPLATE[key].keys():
            raise ValueError(f"Unknown fields in contact_setup.{key}")
    capabilities = (setup.get("instrumentation") or {}).get("capabilities") or {}
    if not isinstance(capabilities, dict) or set(capabilities) - SIGNALS.keys():
        raise ValueError("Unknown signal capabilities")
    if any(value not in {"available", "unavailable", "unknown"} for value in capabilities.values()):
        raise ValueError("Signal capability must be available, unavailable or unknown; not a qualification result")
    questions = []
    for key, prompt in (
        (
            "controller",
            "Record the actual deployment control chain, command semantics, rates, gains, limits and source versions.",
        ),
        ("tool", "Record the mounted gripper/fingers, tool/TCP and payload configuration, with source references."),
    ):
        if not inputs.get(key):
            questions.append(
                {
                    "key": key,
                    "question": prompt,
                    "why": "Unknown hardware facts remain unknown; collection planning can proceed without approving motion.",
                    "blocks": "hardware",
                    "example": None,
                }
            )
    # Always expose review, even when arbitrary JSON has been supplied. Presence
    # of a declaration is not hardware qualification or permission to run.
    questions.append(
        {
            "key": "contact_setup",
            "question": "Review the setup and available instruments with the robot engineer; fill the contact_setup template.",
            "why": "Numeric conditions and limits require local approval. Enter unavailable for absent sensors; do not fabricate values.",
            "blocks": "hardware",
            "example": deepcopy(SETUP_TEMPLATE),
        }
    )
    coverage = []
    for signal in sorted({s for trial in definition["trials"] for s in trial["requires"]}):
        coverage.append(
            {"signal": signal, "declared_availability": capabilities.get(signal, "unknown"), "qualified": False}
        )
    return {
        "questions": questions,
        "proposals": {"evidence_coverage": coverage},
        "usd": {
            "path": session["asset"],
            "sha256": session["asset_sha256"],
            "inspection": "Not inspected by collection-only intake",
        },
        "environment": None,
        "collection_only": True,
    }


def _merge_template(template, supplied):
    result = deepcopy(template)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_template(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _render_checklist(definition, plan):
    esc = lambda value: html.escape(str(value), quote=True)

    def bullets(values):
        return "<ul>" + "".join(f"<li>{esc(value)}</li>" for value in values) + "</ul>"

    rows = "".join(
        f"<tr><td>{esc(t['code'])}</td><td>{esc(t['motion'])}</td><td>{t['repeats']}</td><td>{esc(t['split'])}<br>{esc(t['priority'])}</td><td>{esc(', '.join(t['requires']))}</td></tr>"
        for t in definition["trials"]
    )
    parameters = "".join(
        f"<tr><td>{esc(p['id'])}<br>{esc(p['system'])}</td><td>{esc(p['evidence'])}</td><td>{esc(p['limit'])}</td></tr>"
        for p in definition["parameters"]
    )
    signals = "".join(
        f"<tr><td>{esc(key)}</td><td>{esc(value['record'])}</td><td>{esc(value['rule'])}</td></tr>"
        for key, value in SIGNALS.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'"><title>{esc(definition["title"])}</title><style>
body{{font:17px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#172035;background:#fff}}h1{{font-size:32px}}h2{{margin-top:32px;font-size:23px}}.notice{{border-left:5px solid #76b900;background:#f1f6e9;padding:14px 18px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #dce0e4;padding:12px 8px;overflow-wrap:anywhere}}th{{background:#eef3f8}}.table-wrap{{overflow-x:auto}}code{{overflow-wrap:anywhere}}@media print{{body{{margin:0;font-size:11pt}}table{{font-size:10pt}}h2{{break-after:avoid}}tr{{break-inside:avoid}}.table-wrap{{overflow:visible}}}}
</style></head><body><p>NEWTON CALIBRATION · LAB COLLECTION PROPOSAL</p><h1>{esc(definition["title"])}</h1><p>{esc(definition["purpose"])}</p>
<div class="notice"><strong>Collection specification only.</strong> No robot commands, numeric trajectories, simulation screening, fitting or transfer validation have run. All limits and condition values require review by the responsible operator.</div>
<h2>1. Start with one pilot</h2><p>{esc(definition["pilot"])}</p><p>Reuse existing recordings where their setup and signals are qualified. This kit has not audited uploaded evidence, so the run sheet is not a blanket recollection request.</p>
<h2>2. Bring / confirm</h2>{bullets(definition["minimum_kit"])}{bullets(SETUP_QUESTIONS.values())}
<h2>3. Record these trials after operator review</h2><p>{plan["core_trial_count"]} core pilot trials; {plan["conditional_trial_count"]} conditional identification trials. These are starter counts, not a statistical guarantee or proof that every parameter is identifiable. Do not tune on held-out trials. Treat repeated video frames as samples, not independent trials.</p><div class="table-wrap"><table><thead><tr><th>ID</th><th>Motion family</th><th>Repeats</th><th>Use</th><th>Record</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>4. Match evidence to parameters</h2><div class="table-wrap"><table><thead><tr><th>Parameter / system</th><th>Evidence</th><th>What it does not establish</th></tr></thead><tbody>{parameters}</tbody></table></div>
<h2>5. Save native logs and video for every attempt</h2><div class="table-wrap"><table><thead><tr><th>Signal</th><th>What to save</th><th>Qualification needed</th></tr></thead><tbody>{signals}</tbody></table></div>
<p>Preserve commands and feedback at their native rates and original clocks. Do not resample force to camera rate or fill missing data with zeros. Log failures, aborts, missing sensors and reasons for skipped trials.</p>
<h2>6. Before leaving the lab</h2>{bullets(["Replay the pilot and at least one final recording; verify visible contact/slip and complete logs.", "Check sensor presence, clipping, timestamp gaps, repeated/stale readings and sync uncertainty.", "Capture setup/controller/gripper/tool configuration, dimensions, software versions and source hashes.", "Save the actual trial order and protocol changes; holdouts are whole untouched episodes.", "Back up originals and keep derived signals separate. Do not call a template collected evidence."])}
<h2>Outcomes to label</h2>{bullets(definition["metrics_to_record"])}<h2>Conditional follow-up</h2>{bullets([f"{x['id']}: {x['when']}. {x['collect']}" for x in definition["conditional_experiments"]])}<h2>Limitations</h2>{bullets(definition["limitations"])}
<p>Recipe <code>{esc(definition["id"])}</code> · SHA-256 <code>{esc(definition["_sha256"])}</code><br>Session revision {plan["session_revision"]} · recorded values come from real run records, not recipe expectations.</p></body></html>"""


def prepare_collection_bundle(directory, session, definition):
    """Write an immutable proposal; repeat calls verify, never overwrite edits."""
    root = Path(directory).resolve()
    if root.exists():
        return verify_collection_bundle(root)
    intake = inspect_contact_inputs(session, definition)
    root.mkdir(parents=True, exist_ok=False)
    setup = _merge_template(SETUP_TEMPLATE, session["inputs"].get("contact_setup") or {})
    recipe_snapshot = {key: value for key, value in definition.items() if not key.startswith("_")}
    write_json(root / "recipe.snapshot.json", recipe_snapshot)
    write_json(
        root / "setup.to_review.json",
        {
            "controller": session["inputs"].get("controller"),
            "tool": session["inputs"].get("tool"),
            "contact_setup": setup,
        },
    )
    write_json(
        root / "signal_contract.json",
        {"schema": "newton.contact-signals/v1", "signals": SIGNALS, "ingestion_implemented": False},
    )
    rows = []
    trial_ids = []
    for family in definition["trials"]:
        for repeat in range(1, family["repeats"] + 1):
            trial_id = f"{definition['stage'].upper()}_{family['code']}_r{repeat}"
            trial_ids.append(trial_id)
            row = {
                "trial_id": trial_id,
                "split": family["split"],
                "priority": family["priority"],
                "motion_family": family["motion"],
                "condition": family["condition"],
                "status": "not_collected",
                "actual_order": "",
                "outcome": "",
                "skip_or_abort_reason": "",
            }
            rows.append(row)
            manifest = {
                "schema": "newton.contact-trial-template/v1",
                "recipe_id": definition["id"],
                "recipe_sha256": definition["_sha256"],
                "session_id": session["session_id"],
                "session_revision": session["revision"],
                **row,
                "captured": False,
                "operator": None,
                "recorded_at": None,
                "condition_values_and_units": None,
                "controller_snapshot": None,
                "upstream_packages": None,
                "streams": {
                    s: {"files": [], "source_field": None, "unit_frame_clock": None, "status": "not_recorded"}
                    for s in family["requires"]
                },
                "events": [],
                "outcome_measurements": None,
                "hardware_execution_authorized": False,
            }
            write_json(root / "trial_templates" / family["split"] / trial_id / "manifest.template.json", manifest)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(root / "run_sheet.template.csv", csv_buffer.getvalue())
    plan = {
        "schema": "newton.contact-collection-plan/v1",
        "created_at": utc_now(),
        "recipe_id": definition["id"],
        "recipe_sha256": definition["_sha256"],
        "session_id": session["session_id"],
        "session_revision": session["revision"],
        "asset": session["asset"],
        "asset_sha256": session["asset_sha256"],
        "goal": session["goal"],
        "mode": "collection_only",
        "scientific_calls": {key: "not_run" for key in ("analyze", "plan", "fit", "validate", "write")},
        "template_trials": trial_ids,
        "core_trial_count": sum(t["repeats"] for t in definition["trials"] if t["priority"] == "core"),
        "conditional_trial_count": sum(t["repeats"] for t in definition["trials"] if t["priority"] == "conditional"),
        "evidence_coverage": intake["proposals"]["evidence_coverage"],
        "existing_evidence": session["inputs"].get("evidence"),
        "evidence_audit": "not_implemented_for_contact_recipes",
        "numeric_motion_files": [],
        "preview": {
            "status": "not_supported",
            "reason": "No qualified contact motion generator or runtime preview is installed for this recipe.",
        },
        "hardware_execution_authorized": False,
        "fit_allowed": False,
        "activation_allowed": False,
        "parameter_bindings": "not_qualified_for_installed_Newton_backend",
        "upstream_requirement": "Record current setup now; verify and lock upstream packages or separately qualify upstream response before staged fitting.",
        "validation": {
            "split": "Whole independent trials; never adjacent frames. Holdouts not used for optimization or model selection.",
            "minimum_repeats": "Pilot proposal only; expand based on repeatability and desired evaluation precision.",
            "transfer": "Separate predeclared real-policy evaluation required; no transfer result produced.",
        },
    }
    write_json(root / "collection_plan.json", plan)
    atomic_write_text(root / "LAB_CHECKLIST.html", _render_checklist(definition, plan))
    write_json(
        root / "bundle.json",
        {
            "schema": "newton.contact-collection-bundle/v1",
            "session_id": session["session_id"],
            "session_revision": session["revision"],
            "recipe_sha256": definition["_sha256"],
            "files": {
                file.relative_to(root).as_posix(): sha256_file(file)
                for file in sorted(root.rglob("*"))
                if file.is_file()
            },
        },
    )
    return plan


def verify_collection_bundle(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "bundle.json").read_text())
    if manifest.get("schema") != "newton.contact-collection-bundle/v1" or not manifest.get("files"):
        raise ValueError("Incomplete contact collection bundle; use a new session revision")
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                "Collection proposal changed; copy templates to a separate recording folder and submit inputs as a new revision"
            )
    return json.loads((root / "collection_plan.json").read_text())
