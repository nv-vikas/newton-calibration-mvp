import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from newton_calibration.controllers import (
    ControllerProfile,
    controller_report,
    discover_controller,
    prompt_controller,
    resolve_controller,
)
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import ArticulationEnvCfg, tuning


def action(class_name, **kwargs):
    # CPU contract double with the same MRO as Isaac Lab configclasses. This
    # does not claim an actual Isaac Lab physics/controller integration run.
    cls = type(class_name, (SimpleNamespace,), {"__module__": "isaaclab.envs.mdp.actions.actions_cfg"})
    return cls(asset_name="robot", **kwargs)


def task(kind="OperationalSpaceControllerActionCfg"):
    return SimpleNamespace(
        actions=SimpleNamespace(arm_action=action(kind, controller_cfg=SimpleNamespace(target_types=["pose_rel"]))),
        sim=SimpleNamespace(dt=0.01),
        decimation=3,
        scene=SimpleNamespace(robot=SimpleNamespace(actuators={"arm": {"stiffness": 0.0, "damping": 0.0}})),
    )


def joint_profile(**changes):
    payload = {
        "name": "test-joint-pd-v1",
        "simulation": {
            "kind": "joint_position_pd",
            "implementation": "Isaac Lab IdealPD",
            "source": "test controller config",
            "command_space": "joint_position",
            "frame": "joint",
            "units": "rad",
            "command_rate_hz": 100,
        },
        "real": {
            "interface": "test_driver",
            "mode": "joint_position",
            "source": "operator config",
            "command_space": "joint_position",
            "frame": "joint",
            "units": "rad",
            "command_rate_hz": 100,
        },
        "simulation_confirmed": True,
        "real_confirmed": True,
    }
    payload.update(changes)
    return ControllerProfile(**payload)


@pytest.fixture
def env(tmp_path):
    asset = tmp_path / "arm.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\ndef PhysicsRevoluteJoint "a" {}\n}\n'
    )
    return ArticulationEnvCfg(usd_path=str(asset), joint_groups={"arm": ("a",)}, joint_map={"a": "a"})


def test_live_config_osc_discovery_preserves_settings_and_does_not_confirm():
    cfg = task()
    found = discover_controller(SimpleNamespace(cfg=cfg))
    assert found.simulation["kind"] == "cartesian_osc"
    assert found.simulation["command_space"] == "cartesian_delta"
    assert found.simulation["command_rate_hz"] == pytest.approx(100 / 3)
    assert found.simulation["frame"] == ""  # do not guess task frame
    assert found.simulation["settings"]["actuators"]["arm"]["stiffness"] == 0
    assert not found.simulation_confirmed and not found.real_confirmed
    assert not found.real and cfg.actions.arm_action.controller_cfg.target_types == ["pose_rel"]
    report = controller_report(found)
    assert not report["fit_supported"] and not report["joint_motion_proposals_allowed"]
    assert any(q["id"] == "real_details" for q in report["questions"])


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("JointPositionActionCfg", "joint_position"),
        ("RelativeJointPositionActionCfg", "joint_position_delta"),
        ("JointVelocityActionCfg", "joint_velocity"),
        ("JointEffortActionCfg", "joint_effort"),
        ("DifferentialInverseKinematicsActionCfg", "cartesian_ik"),
    ],
)
def test_action_families_and_downstream_subclasses(kind, expected):
    cfg = task(kind)
    cfg.actions.arm_action = type("CustomDeployAction", (type(cfg.actions.arm_action),), {})(
        **vars(cfg.actions.arm_action)
    )
    found = discover_controller(cfg)
    assert found.simulation["kind"] == expected
    # Discovery alone never certifies the actual replay boundary.
    assert not controller_report(found)["fit_ready"]


def test_no_controller_inference_from_task_name_usd_or_untrusted_class_name():
    cfg = SimpleNamespace(task="insertion", usd_path="arm.usd")
    assert discover_controller(cfg).simulation["kind"] == "unknown"
    fake = type("OperationalSpaceControllerActionCfg", (SimpleNamespace,), {})
    cfg.actions = SimpleNamespace(arm=fake(asset_name="robot"))
    assert discover_controller(cfg).simulation["kind"] == "unknown"


def test_multiple_actions_require_selection_and_preserve_gripper():
    cfg = task()
    cfg.actions.gripper = action("JointPositionActionCfg")
    assert discover_controller(cfg).simulation["kind"] == "ambiguous"
    selected = discover_controller(cfg, action_name="arm_action")
    assert selected.simulation["kind"] == "cartesian_osc"
    assert "gripper" in selected.simulation["settings"]["all_actions"]
    with pytest.raises(ValueError, match="not found"):
        discover_controller(cfg, action_name="missing")


def test_json_roundtrip_fingerprint_and_no_confirmation_by_string(tmp_path):
    profile = joint_profile()
    file = tmp_path / "controller.json"
    file.write_text(json.dumps(profile.to_dict()))
    assert ControllerProfile.load(file).fingerprint == profile.fingerprint
    assert controller_report(profile)["fit_ready"]
    assert replace(profile, real_confirmed=False).fingerprint != profile.fingerprint
    with pytest.raises(TypeError, match="boolean"):
        replace(profile, real_confirmed="yes")
    with pytest.raises(ValueError, match="fields"):
        replace(profile, real={"executable": "do-not-run.py"})


@pytest.mark.parametrize("rate", [0, -1, float("nan"), True, "100"])
def test_invalid_rates_rejected(rate):
    profile = joint_profile()
    with pytest.raises(ValueError, match="positive and finite"):
        replace(profile, real={**profile.real, "command_rate_hz": rate})


def test_live_change_or_falsely_relabelled_controller_rejects_stale_confirmation(env):
    cfg = task()
    detected = discover_controller(cfg)
    profile = replace(detected, simulation_confirmed=True)
    resolved, report = resolve_controller(env.describe(), profile=profile, isaaclab_env=cfg)
    assert resolved.controller_profile == profile.to_dict()
    assert report["kind"] == "cartesian_osc"
    cfg.scene.robot.actuators["arm"]["stiffness"] = 200
    with pytest.raises(ValueError, match="new revision"):
        resolve_controller(env.describe(), profile=profile, isaaclab_env=cfg)
    with pytest.raises(ValueError, match="new revision"):
        resolve_controller(env.describe(), profile=joint_profile(), isaaclab_env=cfg)


def test_confirmed_unsupported_remains_unsupported():
    profile = joint_profile()
    profile = replace(
        profile, simulation={**profile.simulation, "kind": "cartesian_osc", "command_space": "cartesian_delta"}
    )
    report = controller_report(profile)
    assert not report["fit_ready"] and not report["joint_motion_proposals_allowed"]
    assert any("command spaces differ" in message for message in report["blockers"])


@pytest.mark.parametrize("field,value", [("frame", "base"), ("units", "deg"), ("command_rate_hz", 50)])
def test_incompatible_commands_cannot_fit_or_generate(field, value):
    profile = joint_profile()
    report = controller_report(replace(profile, real={**profile.real, field: value}))
    assert not report["fit_ready"] and not report["joint_motion_proposals_allowed"]
    assert any(field in text for text in report["blockers"])


def test_osc_analysis_persists_questions_and_plan_never_runs_pd(env, tmp_path):
    def should_not_run(*args):
        pytest.fail("Unsupported controller called a joint-PD probe/preview")

    analysis = tuning.analyze(env=env, isaaclab_env=task(), workdir=tmp_path)
    assert not analysis.readiness["controller_supported"]
    assert analysis.controller["kind"] == "cartesian_osc"
    stored = json.loads((Path(analysis.workdir) / "controller_discovery.json").read_text())
    assert stored["fingerprint"] == analysis.controller["fingerprint"]
    plan = tuning.plan(analysis, preview=should_not_run, design_probe=should_not_run)
    assert plan.status == "controller_action_required"
    assert not plan.episodes and plan.preview["status"] == "blocked"
    assert not (Path(plan.workdir) / "commands").exists()
    assert plan.assistance["controller"]["discovery"]["questions"]


def test_surface_can_supply_controller_config_without_new_call(env, tmp_path):
    class Surface:
        def describe(self):
            return env.describe()

        def describe_controller_config(self):
            return task()

    plan = tuning.assist(env=Surface(), video=False, workdir=tmp_path)
    assert plan.status == "controller_action_required"


def test_plan_cannot_change_controller_after_analyze(env, tmp_path):
    analysis = tuning.analyze(env=env, controller_profile=joint_profile(), workdir=tmp_path)
    analysis.environment.controller_profile["real"]["mode"] = "changed"
    with pytest.raises(ValueError, match="Analysis changed"):
        tuning.plan(analysis)


def test_profile_passes_through_env_description(env):
    original = joint_profile().to_dict()
    updated = replace(env, controller_profile=original)
    described = updated.describe()
    original["real"]["mode"] = "mutated input"
    assert described.controller_profile["real"]["mode"] == "joint_position"


def test_wizard_does_not_confirm_blank_answers_or_claim_support():
    profile = joint_profile(simulation_confirmed=False, real_confirmed=False)
    answers = iter(["y", ""])
    reviewed = prompt_controller(profile, ask=lambda _: next(answers), show=lambda _: None)
    assert reviewed.simulation_confirmed and not reviewed.real_confirmed
    assert not controller_report(reviewed)["fit_ready"]


def test_cli_writes_unknown_profile_and_machine_readable_questions(tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main

    output = tmp_path / "controller.json"
    monkeypatch.setattr("sys.argv", ["newton-calibration", "controller", "--output", str(output)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["questions"] and not result["fit_ready"]
    assert ControllerProfile.load(output).name == "controller-profile"
    previous = output.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert output.read_bytes() == previous


def test_cli_assist_honors_profile_file(env, tmp_path, monkeypatch, capsys):
    from newton_calibration.cli import main

    config = tmp_path / "job.json"
    config.write_text(json.dumps({"environment": jsonable(env.describe())}))
    profile = tmp_path / "controller.json"
    profile.write_text(json.dumps(discover_controller(task()).to_dict()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "newton-calibration",
            "assist",
            "--config",
            str(config),
            "--controller-profile",
            str(profile),
            "--workdir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "controller_action_required"
    assert result["assistance"]["controller"]["discovery"]["kind"] == "cartesian_osc"


def test_profile_evidence_binding_gates_analysis(env, tmp_path):
    from newton_calibration.core import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files

    episodes = []
    for index, split in enumerate(("train", "heldout")):
        path = tmp_path / f"{split}.csv"
        rows = ["time_s,field,value"]
        for step in range(20):
            for signal in ("command_q", "actual_q", "actual_dq"):
                rows.append(f"{step / 100},a/{signal},{0.001 * (step + index)}")
        path.write_text("\n".join(rows) + "\n")
        episodes.append({"name": split, "path": path.name, "split": split, "trial_id": split})
    evidence = bind_evidence_files(
        root=tmp_path,
        episodes=episodes,
        schema=LongFormSchema(time_column="time_s", time_unit="s"),
        joint_bindings=[JointBinding("a", "a", "rad", "rad", transform_confirmed=True)],
        signal_bindings=[SignalBinding(s, s) for s in ("command_q", "actual_q", "actual_dq")],
    )
    profile = joint_profile()
    missing = tuning.analyze(env=env, controller_profile=profile, evidence=evidence, workdir=tmp_path / "runs")
    assert not missing.readiness["controller_evidence_matches"]
    assert any(q["id"] == "evidence_controller" for q in missing.controller["questions"])
    matched = replace(evidence, controller_profile_fingerprint=profile.fingerprint)
    assert matched.fingerprint != evidence.fingerprint
    ok = tuning.analyze(env=env, controller_profile=profile, evidence=matched, workdir=tmp_path / "runs")
    assert ok.readiness["controller_evidence_matches"]
    wrong = tuning.analyze(
        env=env,
        controller_profile=replace(profile, name="another revision"),
        evidence=matched,
        workdir=tmp_path / "runs",
    )
    assert not wrong.readiness["controller_evidence_matches"]
    # Omitting the controller is not a bypass for explicitly bound evidence.
    omitted = tuning.analyze(env=env, evidence=matched, workdir=tmp_path / "runs")
    assert not omitted.readiness["controller_evidence_matches"]


def test_legacy_environment_fingerprint_has_no_new_empty_field(env):
    assert "controller_profile" not in jsonable(env.describe())


def test_runtime_rejects_unsupported_profile_before_importing_newton(env):
    from newton_calibration.adapters.runtime import create_runtime

    environment = replace(env, controller_profile=discover_controller(task()).to_dict()).describe()
    with pytest.raises(ValueError, match="unsupported or unconfirmed"):
        create_runtime(environment)


def test_matching_but_non_radian_profile_is_not_supported():
    profile = joint_profile()
    report = controller_report(
        replace(profile, simulation={**profile.simulation, "units": "deg"}, real={**profile.real, "units": "deg"})
    )
    assert not report["fit_supported"]


def test_position_action_is_not_assumed_to_have_pd_actuators():
    cfg = task("JointPositionActionCfg")
    assert discover_controller(cfg).simulation["kind"] == "joint_position"
    ideal = type("IdealPDActuatorCfg", (SimpleNamespace,), {"__module__": "isaaclab.actuators.actuator_cfg"})
    cfg.scene.robot.actuators = {"arm": ideal(stiffness=20.0, damping=1.0)}
    found = discover_controller(cfg)
    assert found.simulation["kind"] == "joint_position_pd"
    assert not controller_report(found)["fit_supported"]  # raw policy actions, not downstream targets
    sim = dict(found.simulation)
    sim["settings"] = {**sim["settings"], "evidence_boundary": "postprocessed_joint_targets"}
    assert controller_report(replace(found, simulation=sim))["fit_supported"]


def test_collection_rate_must_match_profile(env, tmp_path):
    from newton_calibration.collection import MotionSpec

    motion = MotionSpec(
        ("a",), (0.0,), (-2.0,), (2.0,), (0.2,), (0.3,), (0.6,), source="test", scene_id="test", command_rate_hz=50
    )
    result = tuning.assist(
        env=env, controller_profile=joint_profile(), collection=motion, video=False, workdir=tmp_path
    )
    assert result.status == "controller_action_required"
    assert not result.episodes and "command rate differs" in result.preview["reason"]
