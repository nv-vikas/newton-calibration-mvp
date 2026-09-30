"""Inspect declared inputs, ask targeted questions, never invent confirmations."""

import math
from pathlib import Path

from newton_calibration.adapters.evidence import TabularJointEvidence
from newton_calibration.adapters.evidence.tabular_joint import inspect_tabular_evidence
from newton_calibration.core.evidence_spec import BoundEvidenceSpec, LongFormSchema, SignalBinding
from newton_calibration.core.joint_mapping import JointBinding
from newton_calibration.core.models import EnvironmentSpec, jsonable
from newton_calibration.isaaclab import tuning
from newton_calibration.recipes import get_recipe as execution_recipe

from .catalog import get_recipe, question
from .store import fingerprint, read

CONFIRMATIONS = ("mapping", "controller", "tool", "bounds")
SECTIONS = {"environment", "controller", "tool", "joint_bindings", "evidence", "collection", "request", "fit_budget"}


def confirmation_basis(inputs, name, asset_sha256):
    # Conservatively re-confirm on any substantive setup change. Evidence upload
    # and optimizer budget changes alone do not revoke an unchanged physical setup.
    evidence_map = None
    if name == "mapping" and inputs.get("evidence"):
        try:
            value = inputs["evidence"]
            value = read(value) if isinstance(value, str) else value
            evidence_map = {key: value.get(key) for key in ("joint_bindings", "schema", "signal_bindings")}
        except (ValueError, OSError, TypeError, AttributeError):
            evidence_map = "unreadable"
    return fingerprint(
        {
            "asset": asset_sha256,
            "name": name,
            "evidence_mapping": evidence_map,
            **{key: inputs.get(key) for key in ("environment", "controller", "tool", "joint_bindings", "request")},
        }
    )


def confirmed(session, name):
    value = session["confirmations"].get(name, {})
    return bool(
        value.get("by")
        and value.get("source")
        and value.get("basis") == confirmation_basis(session["inputs"], name, session["asset_sha256"])
    )


def inspect_inputs(session):
    """Static intake and proposals; this is NOT the scientific analyze call."""
    questions, proposals = [], {}
    inputs = session["inputs"]
    usd = tuning.inspect_usd(session["asset"])
    if usd.blockers:
        questions.append(
            question("asset", "Resolve the asset inspection issues.", "; ".join(usd.blockers), blocks="all")
        )
    env_data = dict(inputs.get("environment") or {})
    if not env_data.get("joint_groups") or not env_data.get("joint_map"):
        names = [joint.name for joint in usd.joints if joint.kind == "revolute"]
        # A proposed USD-local map is sufficient for simulation-only collection.
        # It never asserts the real driver's naming, units, signs or zeros.
        env_data["joint_groups"] = {f"joint_{i:02d}": [name] for i, name in enumerate(names, 1)}
        env_data["joint_map"] = {name: name for name in names}
        proposals["joint_map"] = {
            "value": env_data["joint_map"],
            "confirmed": False,
            "source": "USD joint names; real driver coordinates still unknown",
        }
    if not env_data["joint_map"]:
        return {
            "questions": questions,
            "proposals": proposals,
            "environment": None,
            "evidence": None,
            "usd": usd.to_dict(),
        }
    env_data.update(
        asset_path=session["asset"],
        profile_schema="articulation-profile/v1",
        profile_confirmed=confirmed(session, "mapping"),
        controller_profile_confirmed=confirmed(session, "controller"),
        controller_profile_source=(inputs.get("controller") or {}).get("source", ""),
    )
    env_data.setdefault("adapter", "isaaclab_newton")
    request = inputs.get("request") or {}
    targets = request.get("target_parameters") or [
        f"{group}_{kind}"
        for group in env_data["joint_groups"]
        for kind in get_recipe(session["recipe_id"])["guided"]["default_parameter_families"]
    ]
    env_data["tuning_targets"] = tuple(targets)
    env = EnvironmentSpec(**env_data)
    proposals["parameters"] = {
        "value": jsonable(execution_recipe("articulation.position_pd.free_space@3", env).parameters),
        "confirmed": confirmed(session, "bounds"),
        "source": "Declared bounds or installed recipe scale defaults; review before fitting",
    }
    controller = inputs.get("controller") or {}
    required = (
        "simulation_mode",
        "real_mode",
        "command_semantics",
        "command_unit",
        "command_rate_hz",
        "source",
        "filters",
        "gravity_compensation",
    )
    missing = [key for key in required if controller.get(key) in (None, "", "unknown")]
    if missing:
        questions.append(
            question(
                "controller",
                "Which controller and command path will the robot use?",
                "Missing: " + ", ".join(missing),
                blocks="fit; unknown simulation mode also blocks motion generation",
                example={
                    "simulation_mode": "joint_position_pd",
                    "real_mode": "joint_position",
                    "command_semantics": "absolute_joint_position",
                    "command_unit": "rad",
                    "command_rate_hz": 100,
                    "filters": "describe actual filtering or explicitly none",
                    "gravity_compensation": "describe real and sim",
                    "source": "configuration revision or engineer confirmation",
                },
            )
        )
    mode = controller.get("simulation_mode")
    semantics = controller.get("command_semantics")
    unsupported = (
        mode not in (None, "unknown", "joint_position_pd")
        or semantics not in (None, "unknown", "absolute_joint_position")
        or controller.get("real_mode") not in (None, "unknown", "joint_position", "joint_impedance")
        or controller.get("command_unit") not in (None, "unknown", "rad", "deg")
    )
    if unsupported:
        questions.append(
            question(
                "controller_support",
                "This recipe cannot calibrate that controller yet.",
                "Cartesian, OSC, differential IK and torque commands must not be replaced with joint-PD motions.",
                blocks="all",
            )
        )
    rate = controller.get("command_rate_hz")
    if rate in ("", "unknown"):
        rate = None
    if rate is not None and (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0):
        raise ValueError("Controller command_rate_hz must be finite and positive")
    # The current replay adapter applies a command every physics step. Do not
    # claim equivalence for a deployment action rate it cannot yet reproduce.
    if rate and not math.isclose(float(rate), 1 / env.dt, rel_tol=1e-7):
        questions.append(
            question(
                "controller_rate",
                "Match the replay command rate to the declared controller.",
                "Toolkit engineering must qualify replay of the recorded command timing at the declared physics step, including applicable filtering/limiting. Do not change dt merely to hide this mismatch or request more robot data for this software gap.",
                blocks="fit",
            )
        )
    baselines_missing = [
        key
        for key in ("base_stiffness_by_joint", "base_damping_by_joint", "base_effort_limit_by_joint")
        if set(env_data.get(key, {})) != set(env.joint_map)
    ]
    if baselines_missing:
        questions.append(
            question(
                "environment",
                "Supply the simulated controller's per-joint baseline settings.",
                "These are not inferred hardware gains: " + ", ".join(baselines_missing),
            )
        )
    tool = inputs.get("tool") or {}
    if tool.get("kind") not in {"none", "attached"} or tool.get("matches_asset") is not True or not tool.get("source"):
        questions.append(
            question(
                "tool",
                "What gripper/tool and payload are attached, and does the USD match?",
                "Specify none explicitly, or supply the attached model and mass/COM/inertia/mounting references; unknown is allowed but blocks fitting.",
                example={
                    "kind": "attached",
                    "name": "your gripper",
                    "payload": "none",
                    "dynamics_source": "mass/COM/inertia and mounting reference",
                    "matches_asset": None,
                    "source": "engineer or config",
                },
            )
        )
    elif tool["kind"] == "attached" and any(not tool.get(key) for key in ("name", "payload", "dynamics_source")):
        questions.append(
            question(
                "tool",
                "Complete the attached tool configuration.",
                "Need name, payload (or none) and dynamics/mounting source.",
            )
        )
    for name in CONFIRMATIONS:
        if not confirmed(session, name):
            questions.append(
                question(
                    f"confirm.{name}",
                    f"Review and confirm the {name} information.",
                    "Discovered values and proposals are not automatically confirmed. Record who verified them and the source.",
                )
            )
    evidence = None
    evidence_issue = None
    evidence_proposal = None
    if inputs.get("evidence"):
        try:
            value = inputs["evidence"]
            data = read(value) if isinstance(value, str) else value
            if not isinstance(data, dict):
                raise TypeError("Evidence needs a data-only descriptor or bound-evidence JSON object")
            if data.get("adapter") == "tabular_joint.v1":
                evidence = TabularJointEvidence(BoundEvidenceSpec.from_dict(data))
            else:
                # An unbound descriptor can include just training recordings.
                inventory = inspect_tabular_evidence(
                    root=data["root"], episodes=data["episodes"], schema=LongFormSchema(**data["schema"])
                )
                evidence_proposal = {
                    "inventory": inventory.to_dict(),
                    "mapping": tuning.propose_mapping(usd=usd, source_joints=inventory.source_joints).to_dict(),
                }
                bindings = inputs.get("joint_bindings") or []
                if not bindings:
                    raise ValueError(
                        "Real logs inspected; review the proposed joint map and provide units/signs/offsets"
                    )
                typed = tuple(
                    JointBinding(**{**item, "transform_confirmed": confirmed(session, "mapping")}) for item in bindings
                )
                spec = BoundEvidenceSpec(
                    root=str(Path(data["root"]).expanduser().resolve()),
                    revision=data.get("revision", "local"),
                    episodes=inventory.episodes,
                    schema=inventory.schema,
                    joint_bindings=typed,
                    signal_bindings=tuple(SignalBinding(**item) for item in data["signal_bindings"]),
                    clock_synchronized=data.get("clock_synchronized", False),
                    effort_saturation_joints=tuple(data.get("effort_saturation_joints", [])),
                )
                evidence = TabularJointEvidence(spec, inspection_only=True)
            binding_map = {item.source_joint: item.usd_joint for item in evidence.spec.joint_bindings}
            if env.joint_map != binding_map:
                questions.append(
                    question(
                        "environment",
                        "Align the environment joint map with the evidence bindings.",
                        "Mappings disagree; no automatic remapping or fitting.",
                        blocks="all",
                    )
                )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            evidence_issue = str(exc)
            questions.append(
                question(
                    "evidence",
                    "Help interpret the existing recordings.",
                    evidence_issue,
                    blocks="evidence analysis; existing data will not be silently replaced",
                )
            )
    if evidence_proposal:
        proposals["evidence"] = evidence_proposal
    return {
        "questions": questions,
        "proposals": proposals,
        "environment": env,
        "evidence": evidence,
        "evidence_issue": evidence_issue,
        "usd": usd.to_dict(),
        "unsupported_controller": unsupported,
        "motion_supported": mode == "joint_position_pd" and semantics == "absolute_joint_position" and rate is not None,
    }
