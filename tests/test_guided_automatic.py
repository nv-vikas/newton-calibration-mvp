"""Automatic continuation contracts. Synthetic analytic runs, not Newton proof."""

import json
from copy import deepcopy

import pytest
from test_guided_workflow import create
from test_guided_workflow import setup as setup  # noqa: PLC0414 -- explicit pytest fixture re-export

from newton_calibration.guided import provide, run, status
from newton_calibration.guided.cli import describe
from newton_calibration.isaaclab import tuning


def test_prepares_bounds_and_executes_all_five_calls_once(setup, tmp_path, monkeypatch):
    asset, answers = setup
    answers["confirm"].remove("bounds")
    root = create(tmp_path, asset, answers)
    calls = []
    for name in ("analyze", "plan", "fit", "validate", "write"):
        function = getattr(tuning, name)

        def track(*args, _name=name, _function=function, **kwargs):
            calls.append(_name)
            return _function(*args, **kwargs)

        monkeypatch.setattr(tuning, name, track)
    result = run(root, execute=True)
    assert calls == ["analyze", "plan", "fit", "validate", "write"]
    assert result["state"] == "completed"
    assert result["continuation"]["outcome"] == "completed"
    assert result["continuation"]["user_requests"] == []
    assert result["confirmations"]["bounds"]["reviewer_kind"] == "agent"
    assert result["continuation"]["automatic_preparation"][0]["actions"] == ["reviewed_simulation_search_bounds"]
    assert not result["hardware_execution_authorized"]
    assert not result["real_robot_commands_sent"]
    assert not result["action_plan"]["user_attention_required"]
    again = run(root, execute=True)
    assert again["state"] == "completed"
    assert calls == ["analyze", "plan", "fit", "validate", "write"]
    assert len(again["continuation_history"]) == 2
    assert again["continuation_history"][0] == result["continuation"]


def test_execution_choice_once_and_resume(setup, tmp_path, monkeypatch):
    root = create(tmp_path, *setup)
    fit = tuning.fit
    monkeypatch.setattr(tuning, "fit", lambda *a, **k: pytest.fail("Preparation is not fitting permission"))
    result = run(root)
    assert result["state"] == "ready_to_fit"
    assert [r["key"] for r in result["continuation"]["user_requests"]] == ["execution"]
    plan = result["artifacts"]["plan"]
    monkeypatch.setattr(tuning, "fit", fit)
    result = run(root, execute=True)
    assert result["state"] == "completed" and result["artifacts"]["plan"] == plan


def test_missing_hardware_facts_are_not_invented(setup, tmp_path):
    asset, answers = setup
    answers.pop("confirm")
    root = create(tmp_path, asset, answers)
    result = run(root, execute=True)
    assert set(result["confirmations"]) == {"bounds"}
    assert result["continuation"]["outcome"] == "awaiting_user"
    assert {r["key"] for r in result["continuation"]["user_requests"]} == {
        "confirm.mapping",
        "confirm.controller",
        "confirm.tool",
    }
    assert "fit" not in result["artifacts"]
    output = describe(result)
    assert "confirm.bounds" not in output and "Done when:" not in output
    assert "confirm.controller" in output


def test_supplied_profile_fills_only_missing_maps_with_provenance(setup, tmp_path):
    asset, answers = setup
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(answers["environment"]))
    answers.pop("confirm")
    del answers["controller"]["simulation_mode"]
    del answers["environment"]["base_damping_by_joint"]
    answers["environment"]["base_stiffness_by_joint"]["a"] = 15.0
    root = create(tmp_path, asset, answers)
    original_revision = status(root)["revision"]
    result = run(root, execute=True, simulation_profile=profile)
    assert result["inputs"]["environment"]["base_damping_by_joint"] == {"a": 1.0, "b": 1.0}
    assert result["inputs"]["environment"]["base_stiffness_by_joint"]["a"] == 15.0
    assert result["inputs"]["controller"]["simulation_mode"] == "joint_position_pd"
    assert result["inputs"]["controller"]["real_mode"] == "joint_position"
    assert "sha256:" in result["continuation"]["automatic_preparation"][0]["source"]
    assert result["revision"] > original_revision
    assert (root / "revisions" / f"{original_revision:04d}.json").exists()
    assert "fit" not in result["artifacts"]


def test_unsupported_cartesian_mode_is_toolkit_work_not_user_to_do(setup, tmp_path):
    asset, answers = setup
    answers["controller"].update(
        simulation_mode="cartesian_osc", real_mode="cartesian_impedance", command_semantics="delta_cartesian_pose"
    )
    root = create(tmp_path, asset, answers)
    result = run(root, execute=True)
    assert result["inputs"]["controller"] == answers["controller"]
    assert "fit" not in result["artifacts"]
    assert result["continuation"]["outcome"] == "toolkit_attention"
    assert result["action_plan"]["user_requests"] == []
    assert any(a["id"] == "qualify_replay" for a in result["action_plan"]["items"])
    assert "No background job is running" in describe(result)
    assert "Fix or qualify" not in describe(result)


def test_trusted_resolver_prepares_then_continues_without_manual_advances(setup, tmp_path):
    asset, answers = setup
    descriptor = deepcopy(answers["evidence"])
    answers["evidence"] = {"uninterpreted": "fixture"}
    answers.pop("confirm")
    root = create(tmp_path, asset, answers)
    seen = []

    def resolve(snapshot):
        seen.append(snapshot["revision"])
        if snapshot["inputs"]["evidence"] != descriptor:
            return {"answers": {"evidence": descriptor}, "source": "Synthetic source schema and unit mapping"}
        return None

    result = run(root, execute=True, resolver=resolve)
    assert seen and result["analysis_status"] == "completed"
    assert result["inputs"]["evidence"] == descriptor
    assert any(a["answer_keys"] == ["evidence"] for a in result["continuation"]["automatic_preparation"])
    assert "fit" not in result["artifacts"]  # real mapping remains unconfirmed


@pytest.mark.parametrize(
    "answers",
    [
        {"fit_budget": {"generations": 1000}},
        {"recipe": "arm_joint_response@1"},
        {"request": {"target_parameters": ["new_parameter"]}},
        {"confirm": ["tool"]},
        {"environment": {"adapter": "isaaclab_newton"}},
        {"environment": None},
    ],
)
def test_resolver_cannot_escalate_scope_budget_or_hardware_confirmation(setup, tmp_path, answers):
    asset, original = setup
    original.pop("confirm")
    root = create(tmp_path, asset, original)
    with pytest.raises(ValueError):
        run(root, resolver=lambda s: {"answers": answers, "source": "invalid automatic proposal"})
    latest = status(root)
    assert all(latest["inputs"][key] == value for key, value in original.items())
    assert not {"tool", "mapping", "controller"} & latest["confirmations"].keys()


def test_noop_resolver_stops_without_revision_churn(setup, tmp_path):
    asset, answers = setup
    answers.pop("confirm")
    root = create(tmp_path, asset, answers)
    result = run(root, resolver=lambda s: {"answers": {"controller": s["inputs"]["controller"]}, "source": "same"})
    assert result["continuation"]["reason"] == "no_progress"
    assert len(result["continuation"]["automatic_preparation"]) == 1  # only the bounds review


def test_stale_automatic_proposals_cannot_overwrite_new_input(setup, tmp_path):
    root = create(tmp_path, *setup)
    revision = status(root)["revision"]
    provide(root, {"fit_budget": {"generations": 3, "population": 4}}, source="new user request")
    with pytest.raises(RuntimeError, match="stale"):
        provide(root, {"tool": None}, source="old proposal", expected_revision=revision)
    assert status(root)["inputs"]["tool"] == setup[1]["tool"]


def test_cli_runs_automatic_path(setup, tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main

    root = create(tmp_path, *setup)
    monkeypatch.setattr("sys.argv", ["newton-calibration", "guide", "run", "--session", str(root), "--execute"])
    main()
    output = capsys.readouterr().out
    assert status(root)["state"] == "completed"
    assert "Run complete" in output and "What to do next" not in output


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execute": "false"},
        {"max_passes": 0},
        {"max_passes": True},
        {"max_passes": 33},
        {"agent_name": " "},
        {"resolver": "recipe.py"},
    ],
)
def test_invalid_automation_controls_are_rejected_before_session_changes(setup, tmp_path, kwargs):
    root = create(tmp_path, *setup)
    before = (root / "session.json").read_bytes()
    with pytest.raises((ValueError, TypeError)):
        run(root, **kwargs)
    assert (root / "session.json").read_bytes() == before


def test_preparation_pass_limit_still_analyzes_last_revision(setup, tmp_path):
    asset, answers = setup
    answers["confirm"].remove("bounds")
    root = create(tmp_path, asset, answers)
    result = run(root, max_passes=1)
    assert result["analysis_status"] == "completed" and result["state"] == "ready_to_fit"
    assert result["continuation"]["reason"] == "pass_limit"
    assert "fit" not in result["artifacts"]


def test_missing_evidence_creates_supported_motions_without_permission_to_fit(setup, tmp_path):
    asset, answers = setup
    answers["evidence"] = None
    root = create(tmp_path, asset, answers)
    result = run(root, execute=True)
    assert result["analysis_status"] == "completed"
    assert result["collection"]["episodes"]
    assert result["collection"]["preview"]["status"] != "complete"  # no installed scene hook in this fixture
    assert result["continuation"]["outcome"] == "toolkit_attention"
    assert "fit" not in result["artifacts"]
    assert not result["hardware_execution_authorized"]
