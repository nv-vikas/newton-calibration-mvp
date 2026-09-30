"""Contract tests: synthetic logs + analytic backend, NOT hardware/Newton proof."""

import csv
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from newton_calibration.guided import advance, list_recipes, provide, review, start, status
from newton_calibration.guided.store import read
from newton_calibration.isaaclab import tuning


@pytest.fixture
def setup(tmp_path):
    asset = tmp_path / "arm.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "shoulder" {}\ndef PhysicsRevoluteJoint "elbow" {}\n}\n'
    )
    data = tmp_path / "evidence"
    data.mkdir()
    for name, phase in (("training", 0.0), ("testing", 0.43)):
        with (data / f"{name}.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "joint", "signal", "value"])
            t = np.arange(0, 2.0, 0.02)
            for joint in ("a", "b"):
                q = 0.4 * np.sin(2 * np.pi * 0.75 * t + phase)
                for signal, values in (
                    ("command_q", q),
                    ("actual_q", q * 0.85),
                    ("actual_dq", np.gradient(q * 0.85, 0.02)),
                ):
                    writer.writerows((float(time), joint, signal, float(value)) for time, value in zip(t, values))
    descriptor = {
        "root": str(data),
        "schema": {
            "time_column": "time",
            "time_unit": "s",
            "value_column": "value",
            "joint_column": "joint",
            "signal_column": "signal",
            "field_column": None,
        },
        "episodes": [
            {"name": name, "path": f"{name}.csv", "split": split, "trial_id": f"capture-{name}"}
            for name, split in (("training", "train"), ("testing", "heldout"))
        ],
        "signal_bindings": [
            {"source_signal": name, "canonical_signal": name} for name in ("command_q", "actual_q", "actual_dq")
        ],
    }
    answers = {
        "environment": {
            "adapter": "analytic",
            "device": "cpu",
            "robot_id": "synthetic-arm",
            "dt": 0.02,
            "joint_map": {"a": "shoulder", "b": "elbow"},
            "joint_groups": {"arm": ["a", "b"]},
            "base_stiffness_by_joint": {"a": 20.0, "b": 20.0},
            "base_damping_by_joint": {"a": 1.0, "b": 1.0},
            "base_effort_limit_by_joint": {"a": 20.0, "b": 20.0},
            "analytic_inertia_by_joint": {"a": 1.0, "b": 1.0},
        },
        "controller": {
            "simulation_mode": "joint_position_pd",
            "real_mode": "joint_position",
            "command_semantics": "absolute_joint_position",
            "command_unit": "rad",
            "command_rate_hz": 50,
            "source": "synthetic test config",
            "filters": "none",
            "gravity_compensation": "test only",
        },
        "tool": {"kind": "none", "matches_asset": True, "source": "synthetic fixture"},
        "joint_bindings": [
            {
                "source_joint": source,
                "usd_joint": target,
                "source_unit": "rad",
                "usd_unit": "rad",
                "sign": 1,
                "scale": 1.0,
                "offset": 0.0,
            }
            for source, target in (("a", "shoulder"), ("b", "elbow"))
        ],
        "evidence": descriptor,
        "collection": {
            "joint_names": ["shoulder", "elbow"],
            "center_rad": [0.0, 0.0],
            "lower_rad": [-2.0, -2.0],
            "upper_rad": [2.0, 2.0],
            "amplitude_rad": [0.3, 0.3],
            "max_velocity_rad_s": [0.4, 0.4],
            "max_acceleration_rad_s2": [0.6, 0.6],
            "source": "test scene simulation envelope only",
            "scene_id": "fixture-scene",
            "command_rate_hz": 50,
            "duration_s": 12.0,
        },
        "fit_budget": {"generations": 2, "population": 4},
        "confirm": ["mapping", "controller", "tool", "bounds"],
    }
    return asset, answers


def create(tmp_path, asset, answers=None, recipe="arm_joint_response@1"):
    root = tmp_path / "session"
    start(asset=asset, goal="Prepare for insertion; this recipe is free-motion only", directory=root, recipe=recipe)
    if answers is not None:
        provide(root, answers, source="synthetic test operator", confirmed_by="test engineer")
    return root


def test_catalog_truthfully_marks_future_work(setup, tmp_path):
    recipes = list_recipes()
    assert [r["id"] for r in recipes if r["status"] == "available"] == ["arm_joint_response@1"]
    for recipe in ("grasp", "insertion"):
        with pytest.raises(ValueError, match="not available"):
            start(asset=setup[0], goal="task", directory=tmp_path / recipe, recipe=recipe)
        assert not (tmp_path / recipe).exists()


def test_asset_and_goal_only_is_a_guided_start(setup, tmp_path):
    root = create(tmp_path, setup[0], recipe=None)
    first = review(root)
    assert first["state"] == "choose_recipe"
    assert first["questions"][0]["key"] == "recipe"
    provide(root, {"recipe": "arm_joint_response@1"}, source="user choice")
    result = advance(root)
    assert result["analysis_status"] == "completed"
    assert result["state"] == "needs_collection_setup"
    assert not result["real_robot_commands_sent"]
    assert "fit" not in result["artifacts"]
    assert result["proposals"]["joint_map"]["confirmed"] is False


@pytest.mark.parametrize("recipe_given,evidence_given", [(True, True), (False, False), (False, True), (True, False)])
def test_four_entry_cases_share_the_same_workflow(setup, tmp_path, recipe_given, evidence_given):
    asset, answers = setup
    if not evidence_given:
        answers["evidence"] = None
    root = create(tmp_path, asset, recipe="arm_joint_response@1" if recipe_given else None)
    if not recipe_given:
        provide(root, {"recipe": "arm_joint_response@1"}, source="selected by user")
    provide(root, answers, source="test operator", confirmed_by="engineer")
    result = advance(root)
    assert result["analysis_status"] == "completed"
    if evidence_given:
        assert result["state"] == "ready_to_fit"
        assert "collection" not in result["artifacts"]
    else:
        assert result["state"] == "collection_needs_review"
        assert result["collection"]["episodes"]
        assert result["collection"]["preview"]["requested"] is True
        assert "fit" not in result["artifacts"]
    assert not result["hardware_execution_authorized"]


def test_missing_confirmations_never_bypassed(setup, tmp_path, monkeypatch):
    asset, answers = setup
    answers.pop("confirm")
    # Booleans supplied in environment JSON are not confirmations.
    answers["environment"].update(profile_confirmed=True, controller_profile_confirmed=True)
    root = create(tmp_path, asset, answers)
    monkeypatch.setattr(tuning, "fit", lambda *a, **k: pytest.fail("fit must not run"))
    result = advance(root, execute=True)
    assert result["analysis_status"] == "completed"
    assert result["state"] == "needs_information"
    assert not result["fit_allowed"]
    assert "plan" not in result["artifacts"]


def test_cartesian_controller_does_not_get_substitute_pd_motions(setup, tmp_path, monkeypatch):
    asset, answers = setup
    answers["evidence"] = None
    answers["controller"]["simulation_mode"] = "operational_space"
    root = create(tmp_path, asset, answers)
    monkeypatch.setattr(tuning, "plan", lambda *a, **k: pytest.fail("plan must not run"))
    result = advance(root, execute=True)
    assert result["state"] == "needs_information"
    assert any(q["key"] == "controller_support" for q in result["questions"])
    assert not result["artifacts"]


def test_partial_train_data_is_reused_and_only_holdouts_requested(setup, tmp_path):
    asset, answers = setup
    answers["evidence"]["episodes"] = answers["evidence"]["episodes"][:1]
    root = create(tmp_path, asset, answers)
    result = advance(root, execute=True)
    assert result["state"] == "collection_needs_review"
    assert {e["split"] for e in result["collection"]["episodes"]} == {"heldout"}
    assert not result["fit_readiness"]["evidence_contract_ready"]
    assert "fit" not in result["artifacts"]


def test_collection_then_evidence_continues_same_session(setup, tmp_path):
    asset, answers = setup
    evidence = deepcopy(answers["evidence"])
    answers["evidence"] = None
    root = create(tmp_path, asset, answers)
    first = advance(root)
    old_motion = root / first["artifacts"]["collection"]["path"]
    provide(
        root,
        {"evidence": evidence, "confirm": ["mapping"]},
        source="uploaded test captures; map reviewed",
        confirmed_by="engineer",
    )
    second = advance(root)
    assert first["session_id"] == second["session_id"]
    assert second["revision"] > first["revision"]
    assert second["state"] == "ready_to_fit"
    assert old_motion.exists() and second["history"][-1]["artifacts"]["collection"]


def test_five_actual_calls_and_report_no_bypass(setup, tmp_path):
    root = create(tmp_path, *setup)
    result = advance(root, execute=True)
    assert result["state"] == "completed"
    assert set(result["artifacts"]) == {"intake", "analyze", "plan", "fit", "validate", "write", "report"}
    assert result["activation_allowed"] is False  # analytic backend can never activate a package
    assert Path(result["customer_report"]).is_file()
    assert read(root / result["artifacts"]["analyze"]["path"])["environment"]["adapter"] == "analytic"
    bundle = read(root / result["artifacts"]["report"]["path"])
    assert "guided_intake" in bundle["records"]
    before = deepcopy(result["artifacts"])
    again = advance(root, execute=True)
    assert again["artifacts"] == before
    assert advance(root)["state"] == "completed"


def test_new_revision_clears_current_outputs_but_preserves_them_in_history(setup, tmp_path):
    root = create(tmp_path, *setup)
    completed = advance(root, execute=True)
    changed = provide(root, {"evidence": None}, source="prepare a different collection")
    for field in ("customer_report", "analysis_status", "validation_passed", "activation_allowed"):
        assert field not in changed
    assert Path(completed["customer_report"]).is_file()
    assert changed["history"][-1]["artifacts"]["report"] == completed["artifacts"]["report"]
    assert completed["action_plan"]["items"][0]["id"] == "review_result"
    assert changed["action_plan"]["basis_revision"] == changed["revision"]
    assert all(item["id"] != "review_result" for item in changed["action_plan"]["items"])


def test_partial_guided_report_has_actions_and_verified_unrun_stages(setup, tmp_path, monkeypatch):
    from newton_calibration.guided.catalog import ARM_RECIPE
    from newton_calibration.reporting import build_report, render_report
    from newton_calibration.reporting.adapters import export_toolkit_run

    asset, answers = setup
    answers["controller"]["command_rate_hz"] = 30  # fixture physics is 50 Hz
    root = create(tmp_path, asset, answers)
    monkeypatch.setattr(tuning, "fit", lambda *a, **k: pytest.fail("blocked replay must not fit"))
    result = advance(root, execute=True)
    assert result["fit_allowed"] is False and result["state"] == "needs_information"
    assert "Do not change dt" in next(q for q in result["questions"] if q["key"] == "controller_rate")["why"]
    run = (root / result["artifacts"]["analyze"]["path"]).parent
    recipe = ARM_RECIPE
    bundle = export_toolkit_run(run, recipe, tmp_path / "report", session_path=root / "session.json")
    report = build_report(recipe, bundle)
    assert report["values"]["action_plan"]["items"][0]["id"] == "qualify_replay"
    assert report["values"]["action_plan"]["data_collection"]["status"] == "deferred"
    assert all(report["values"][f"{step}.status"] == "not_run" for step in ("plan", "fit", "validate", "write"))
    assert report["values"]["baseline_error"] is None and report["values"]["activation_allowed"] is None
    html = render_report(report, tmp_path / "partial.html").read_text()
    assert "Analysis complete. Review the next actions." in html
    assert "Done when:" in html and "No new robot data requested yet" in html
    assert '<details class="warning">' in html
    no_session = export_toolkit_run(run, recipe, tmp_path / "without-session")
    assert build_report(recipe, no_session)["values"]["fit.status"] is None

    # An attempted step with no result must remain unknown, not "not run".
    from newton_calibration.core.io import atomic_write_json

    checkpoint = read(root / "session.json")
    checkpoint["events"].append({"revision": checkpoint["revision"], "event": "fit:started_or_resumed"})
    snapshot = tmp_path / "attempted.json"
    atomic_write_json(snapshot, checkpoint)
    attempted = export_toolkit_run(run, recipe, tmp_path / "attempted-report", session_path=snapshot)
    assert build_report(recipe, attempted)["values"]["fit.status"] is None


@pytest.mark.parametrize("mutation", ["hash", "revision", "missing_reference", "missing_record"])
def test_reporting_rejects_mismatched_session_checkpoint(setup, tmp_path, mutation):
    from newton_calibration.core.io import atomic_write_json
    from newton_calibration.guided.catalog import ARM_RECIPE
    from newton_calibration.reporting import ReportError
    from newton_calibration.reporting.adapters import export_toolkit_run

    root = create(tmp_path, *setup)
    result = advance(root)
    run = (root / result["artifacts"]["analyze"]["path"]).parent
    checkpoint = read(root / "session.json")
    if mutation == "hash":
        checkpoint["artifacts"]["analyze"]["sha256"] = "0" * 64
    elif mutation == "revision":
        checkpoint["revision"] += 1
    elif mutation == "missing_reference":
        del checkpoint["artifacts"]["plan"]
    else:
        checkpoint["artifacts"]["fit"] = {"sha256": "0" * 64}
    snapshot = tmp_path / "mismatched.json"
    atomic_write_json(snapshot, checkpoint)
    with pytest.raises(ReportError, match="Guided session"):
        export_toolkit_run(run, ARM_RECIPE, tmp_path / "bad-report", session_path=snapshot)


def test_status_can_be_read_during_active_fit(setup, tmp_path):
    from newton_calibration.guided.store import locked, save

    root = create(tmp_path, *setup)
    with locked(root) as (directory, current):
        current["state"] = "fitting"
        save(directory, current, "test:start")
        assert status(root)["state"] == "fitting"
        with pytest.raises(RuntimeError, match="active writer"):
            advance(root)


def test_preview_failure_is_recorded_not_ignored(setup, tmp_path):
    asset, answers = setup
    answers["evidence"] = None
    root = create(tmp_path, asset, answers)
    calls = []

    def unavailable(command_plan, destination):
        calls.append(command_plan)
        raise RuntimeError("Isaac Lab scene is not running")

    result = advance(root, execute=True, preview=unavailable)
    assert len(calls) == 1
    assert result["collection"]["preview"]["status"] == "failed"
    assert result["state"] == "collection_needs_review"
    assert not result["fit_allowed"]
    assert "fit" not in result["artifacts"]


def test_inspection_only_evidence_cannot_be_replayed(setup, tmp_path):
    from newton_calibration.guided.intake import inspect_inputs

    asset, answers = setup
    answers["evidence"]["episodes"] = answers["evidence"]["episodes"][:1]
    root = create(tmp_path, asset, answers)
    evidence = inspect_inputs(status(root))["evidence"]
    with pytest.raises(ValueError, match="[Ii]nspection"):
        evidence.load_episode("training", dt=0.02)


def test_changed_evidence_bytes_invalidate_saved_plan(setup, tmp_path):
    root = create(tmp_path, *setup)
    advance(root)
    path = Path(setup[1]["evidence"]["root"]) / "training.csv"
    path.write_text(path.read_text().replace("0.34", "0.35"))
    with pytest.raises(ValueError, match="Input files changed"):
        advance(root, execute=True)
    assert "fit" not in status(root)["artifacts"]


def test_controller_change_invalidates_confirmation_and_keeps_history(setup, tmp_path):
    root = create(tmp_path, *setup)
    prior = advance(root)
    changed = deepcopy(setup[1]["controller"])
    changed["filters"] = "new filter"
    provide(root, {"controller": changed}, source="new config revision")
    result = advance(root, execute=True)
    assert result["state"] == "needs_information"
    assert any(q["key"] == "confirm.controller" for q in result["questions"])
    assert (root / prior["artifacts"]["plan"]["path"]).exists()
    assert "fit" not in result["artifacts"]


def test_missing_confirmation_identity_rejected(setup, tmp_path):
    root = create(tmp_path, setup[0])
    with pytest.raises(ValueError, match="person"):
        provide(root, {"confirm": ["mapping"]}, source="agent guess")


def test_changed_usd_is_not_silently_reused(setup, tmp_path):
    root = create(tmp_path, *setup)
    advance(root)
    setup[0].write_text(setup[0].read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="USD changed"):
        advance(root, execute=True)
    assert read(root / "session.json")["state"] == "needs_attention"


def test_error_records_failure_and_resume_uses_same_plan(setup, tmp_path, monkeypatch):
    root = create(tmp_path, *setup)
    actual_fit = tuning.fit
    monkeypatch.setattr(
        tuning, "fit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("GPU temporarily unavailable"))
    )
    with pytest.raises(RuntimeError, match="GPU"):
        advance(root, execute=True)
    failed = read(root / "session.json")
    assert failed["state"] == "needs_attention" and failed["error"]["step"] == "fit"
    monkeypatch.setattr(tuning, "fit", actual_fit)
    result = advance(root, execute=True)
    assert result["state"] == "completed"
    assert result["artifacts"]["plan"] == failed["artifacts"]["plan"]


def test_existing_uninterpretable_data_not_treated_as_absent(setup, tmp_path):
    asset, answers = setup
    answers["evidence"] = {"incorrect": "layout"}
    root = create(tmp_path, asset, answers)
    result = advance(root, execute=True)
    assert result["state"] == "needs_evidence_description"
    assert "collection" not in result["artifacts"]
    assert "analyze" not in result["artifacts"]


def test_cli_catalog_and_asset_intake(setup, tmp_path, capsys, monkeypatch):
    from newton_calibration.cli import main

    monkeypatch.setattr("sys.argv", ["newton-calibration", "guide", "recipes", "--json"])
    main()
    assert '"planned"' in capsys.readouterr().out
    monkeypatch.setattr(
        "sys.argv",
        [
            "newton-calibration",
            "guide",
            "start",
            "--asset",
            str(setup[0]),
            "--goal",
            "task",
            "--session",
            str(tmp_path / "cli-session"),
        ],
    )
    main()
    assert "choose recipe" in capsys.readouterr().out


def test_wizard_requires_explicit_permission_to_fit(setup, tmp_path, monkeypatch, capsys):
    from newton_calibration.guided.cli import wizard

    root = create(tmp_path, *setup)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    monkeypatch.setattr(tuning, "fit", lambda *a, **k: pytest.fail("No execution requested"))
    result = wizard(root)
    assert result["state"] == "ready_to_fit"
    assert "No commands have been sent" in capsys.readouterr().out


def test_wizard_accepts_unknown_without_looping(setup, tmp_path, monkeypatch):
    from newton_calibration.guided.cli import wizard

    root = create(tmp_path, setup[0])
    prompts = []

    def unknown(prompt):
        prompts.append(prompt)
        assert len(prompts) < 15
        return "?"

    monkeypatch.setattr("builtins.input", unknown)
    result = wizard(root)
    assert result["state"] == "needs_collection_setup"
    assert not result["fit_allowed"]
    assert not result["confirmations"]
