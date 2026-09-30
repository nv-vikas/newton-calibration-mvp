"""Controller discovery and user-confirmed command contracts; no robot execution.

Discovery reads an already constructed, trusted Isaac Lab config. It never
imports a task from a USD, executes configuration files, or infers a real
robot's control mode from the task name. Only absolute joint-position replay
is supported by MVP1; discovering another controller does not install it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from newton_calibration.core.attestation import record_fingerprint

_KINDS = {
    "unknown",
    "ambiguous",
    "joint_position_pd",
    "joint_position",
    "joint_position_delta",
    "joint_velocity",
    "joint_effort",
    "cartesian_osc",
    "cartesian_ik",
}
_SPACES = {"", "joint_position", "joint_delta", "joint_velocity", "joint_effort", "cartesian_pose", "cartesian_delta"}
_SIM_FIELDS = {"kind", "implementation", "source", "command_space", "frame", "units", "command_rate_hz", "settings"}
_REAL_FIELDS = {"interface", "mode", "source", "command_space", "frame", "units", "command_rate_hz", "settings"}


@dataclass(frozen=True)
class ControllerProfile:
    name: str
    simulation: dict[str, Any] = field(default_factory=dict)
    real: dict[str, Any] = field(default_factory=dict)
    simulation_confirmed: bool = False
    real_confirmed: bool = False
    schema: str = "newton.controller/v1"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Controller profile requires a name")
        if self.schema != "newton.controller/v1":
            raise ValueError("Unsupported controller profile schema")
        for key in ("simulation_confirmed", "real_confirmed"):
            if not isinstance(getattr(self, key), bool):
                raise TypeError(f"{key} must be a boolean")
        for side, allowed in (("simulation", _SIM_FIELDS), ("real", _REAL_FIELDS)):
            value = getattr(self, side)
            if not isinstance(value, dict) or set(value) - allowed:
                raise ValueError(f"Invalid {side} controller fields: expected {sorted(allowed)}")
            for key, item in value.items():
                if key not in {"settings", "command_rate_hz"} and not isinstance(item, str):
                    raise TypeError(f"{side}.{key} must be text")
            if "settings" in value and not isinstance(value["settings"], dict):
                raise TypeError(f"{side}.settings must be an object")
            rate = value.get("command_rate_hz")
            if rate is not None and (
                isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0
            ):
                raise ValueError(f"{side}.command_rate_hz must be positive and finite")
            if value.get("command_space", "") not in _SPACES:
                raise ValueError(f"Unknown {side} command space")
            # Deep copy and reject non-JSON objects / NaN before locking records.
            object.__setattr__(self, side, json.loads(json.dumps(value, allow_nan=False)))
        if self.simulation.get("kind", "unknown") not in _KINDS:
            raise ValueError("Unknown simulation controller kind")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ControllerProfile:
        if not isinstance(value, dict):
            raise TypeError("Controller profile must be an object")
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path) -> ControllerProfile:
        """Read data-only JSON. Python / YAML execution is intentionally absent."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def fingerprint(self) -> str:
        return record_fingerprint(self.to_dict())


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _class_name(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _snapshot(value, depth=0):
    """Snapshot config data without invoking to_dict(), factories or descriptors."""
    if depth > 12:
        raise ValueError("Controller configuration is too deeply nested")
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_snapshot(v, depth + 1) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _snapshot(v, depth + 1) for k, v in value.items()}
    if isinstance(value, type):
        return _class_name(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "__dict__"):
        return {k: _snapshot(v, depth + 1) for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"Cannot fingerprint controller config value {_class_name(value)}")


def _action_kind(action):
    kinds = {
        "OperationalSpaceControllerActionCfg": ("cartesian_osc", "cartesian_delta", "m,rad"),
        "DifferentialInverseKinematicsActionCfg": ("cartesian_ik", "cartesian_delta", "m,rad"),
        "RelativeJointPositionActionCfg": ("joint_position_delta", "joint_delta", "rad"),
        "JointPositionActionCfg": ("joint_position_pd", "joint_position", "rad"),
        "JointVelocityActionCfg": ("joint_velocity", "joint_velocity", "rad/s"),
        "JointEffortActionCfg": ("joint_effort", "joint_effort", "N m"),
    }
    for cls in type(action).__mro__:
        if cls.__module__.startswith("isaaclab.") and cls.__name__ in kinds:
            return kinds[cls.__name__]
    return "unknown", "", ""


def discover_controller(
    isaaclab_env, *, action_name: str | None = None, asset_name: str = "robot"
) -> ControllerProfile:
    """Inspect a live env's cfg or an already constructed Isaac Lab config.

    Multiple action terms require explicit selection; names like 'insertion'
    are never used to guess a controller. Gripper terms remain in the snapshot.
    """
    cfg = _get(isaaclab_env, "cfg", isaaclab_env)
    actions = _get(cfg, "actions")
    terms = actions if isinstance(actions, Mapping) else vars(actions) if actions is not None else {}
    candidates = {
        k: v
        for k, v in terms.items()
        if not k.startswith("_") and v is not None and _get(v, "asset_name") == asset_name
    }
    if action_name is not None and action_name not in candidates:
        raise ValueError(f"Action {action_name!r} not found for asset {asset_name!r}; available: {sorted(candidates)}")
    selected = action_name or (next(iter(candidates)) if len(candidates) == 1 else None)
    sim = {
        "kind": "ambiguous" if candidates else "unknown",
        "source": f"Isaac Lab config: {_class_name(cfg)}",
        "settings": {"asset_name": asset_name, "available_actions": sorted(candidates)},
    }
    if selected:
        action = candidates[selected]
        kind, space, units = _action_kind(action)
        controller_cfg = _get(action, "controller_cfg")
        # Absolute/relative targets are semantic, not just tensor dimensions.
        if kind == "cartesian_osc" and _get(controller_cfg, "target_types") != ["pose_rel"]:
            space = "cartesian_pose" if _get(controller_cfg, "target_types") == ["pose_abs"] else ""
        if kind == "cartesian_ik" and not _get(controller_cfg, "use_relative_mode", False):
            space = "cartesian_pose"
        dt, decimation = _get(_get(cfg, "sim"), "dt"), _get(cfg, "decimation")
        rate = 1 / (dt * decimation) if dt and decimation else None
        robot = _get(_get(cfg, "scene"), asset_name)
        actuators = _get(robot, "actuators", {})
        # An absolute-position action does not establish the actuator model:
        # it may drive implicit, learned, delayed or other custom actuators.
        # Only positively identified IdealPD configs match the replay model.
        if kind == "joint_position_pd" and not (
            actuators
            and all(
                type(a).__module__.startswith("isaaclab.actuators") and type(a).__name__ == "IdealPDActuatorCfg"
                for a in actuators.values()
            )
        ):
            kind = "joint_position"
        sim.update(
            kind=kind,
            implementation=_class_name(action),
            command_space=space,
            frame="joint" if space.startswith("joint") else "",
            units=units,
            command_rate_hz=rate,
        )
        sim["settings"].update(
            action_name=selected,
            action=_snapshot(action),
            actuators=_snapshot(actuators),
            actuator_implementations={k: _class_name(v) for k, v in actuators.items()},
            all_actions={k: _snapshot(v) for k, v in candidates.items()},
            simulation_dt_s=dt,
            policy_decimation=decimation,
        )
    return ControllerProfile(name=f"{asset_name}-controller", simulation=sim)


def controller_report(profile: ControllerProfile | None, *, legacy_confirmed=False, legacy_source="") -> dict[str, Any]:
    """Human/agent questions and deterministic support gates. Never hardware approval."""
    questions = []
    blockers = []
    if profile is None:
        return {
            "status": "legacy_joint_pd" if legacy_confirmed else "needs_controller_confirmation",
            "discovery": "No task controller inspected; existing MVP1 joint-PD replay surface only. USD and task name do not identify a controller.",
            "profile": {},
            "fingerprint": "",
            "kind": "unknown",
            "fit_supported": True,
            "fit_ready": legacy_confirmed and bool(legacy_source),
            "joint_motion_proposals_allowed": True,
            "questions": [
                {
                    "id": "controller_profile",
                    "question": "Which controller will you use? Supply an Isaac Lab environment config or a controller-profile JSON. Existing joint-PD motions remain provisional.",
                }
            ],
            "warnings": ["Legacy compatibility is not discovery or verification of the deployment controller."],
            "real_execution_approved": False,
        }
    sim, real = profile.simulation, profile.real
    kind = sim.get("kind", "unknown")
    supported = kind == "joint_position_pd" and sim.get("command_space") == "joint_position"
    if supported and (sim.get("frame", "joint") not in {"", "joint"} or sim.get("units", "rad") not in {"", "rad"}):
        supported = False
        blockers.append(
            "MVP1 replay expects radians in joint coordinates; convert evidence explicitly before using this adapter."
        )
    if not supported:
        blockers.append(
            f"Controller {kind!r} is discovery-only: MVP1 has no fitting/collection adapter for this control path."
        )
    # A policy action wrapper is not necessarily the absolute setpoint boundary
    # replayed by MVP1. Make that boundary explicit instead of silently bypassing it.
    if (
        sim.get("settings", {}).get("action") is not None
        and sim.get("settings", {}).get("evidence_boundary") != "postprocessed_joint_targets"
    ):
        supported = False
        blockers.append(
            "Declare settings.evidence_boundary=postprocessed_joint_targets only when logs and preview use the downstream absolute joint targets; raw policy actions are unsupported."
        )
    fields = {
        "simulation": ("implementation", "source", "command_space", "frame", "units", "command_rate_hz"),
        "real": ("interface", "mode", "source", "command_space", "frame", "units", "command_rate_hz"),
    }
    for side, names in fields.items():
        data = sim if side == "simulation" else real
        missing = [n for n in names if not data.get(n)]
        if missing:
            questions.append(
                {
                    "id": f"{side}_details",
                    "question": f"Provide {side} controller details: {', '.join(missing)}.",
                    "missing_fields": missing,
                }
            )
        if not getattr(profile, f"{side}_confirmed"):
            questions.append(
                {
                    "id": f"confirm_{side}",
                    "question": f"Confirm the {side} controller configuration and its source. Confirmation is not evidence that sim and real match.",
                }
            )
    if sim.get("command_space") and real.get("command_space") and sim["command_space"] != real["command_space"]:
        blockers.append("Simulation and real command spaces differ; a tested command adapter is required.")
    for key in ("frame", "units", "command_rate_hz"):
        if sim.get(key) and real.get(key) and sim[key] != real[key]:
            blockers.append(f"Simulation and real {key} differ; resolve the conversion/resampling before fitting.")
    return {
        "status": "unsupported_controller"
        if not supported
        else "needs_controller_confirmation"
        if questions or blockers
        else "ready_for_joint_replay",
        "profile": profile.to_dict(),
        "fingerprint": profile.fingerprint,
        "kind": kind,
        "fit_supported": supported,
        "fit_ready": supported and not questions and not blockers,
        "joint_motion_proposals_allowed": supported and not blockers,
        "questions": questions,
        "blockers": blockers,
        "warnings": [
            "Controller confirmation does not prove response equivalence or approve hardware execution. Gains in this profile are provenance, not automatically applied fitting baselines."
        ],
        "real_execution_approved": False,
    }


def prompt_controller(profile: ControllerProfile, *, ask=input, show=print) -> ControllerProfile:
    """Optional CLI wizard. Agents consume report questions instead of stdin.

    Only data is collected here. Saying 'yes' does not enable unsupported
    adapters or approve real robot motion. Existing files are not overwritten.
    """
    payload = profile.to_dict()
    for side in ("simulation", "real"):
        values = payload[side]
        show(f"{side.title()} controller: {json.dumps(values, indent=2)}")
        if side == "simulation" and values.get("kind", "unknown") in {"unknown", "ambiguous"}:
            if values.get("kind") == "ambiguous":
                raise ValueError(
                    "Select --action-name and rediscover; do not manually replace an ambiguous live controller"
                )
            show("Controller kinds: " + ", ".join(sorted(_KINDS - {"unknown", "ambiguous"})))
            values["kind"] = ask("Controller kind (not task name): ").strip()
        names = (
            ("implementation", "source", "command_space", "frame", "units", "command_rate_hz")
            if side == "simulation"
            else ("interface", "mode", "source", "command_space", "frame", "units", "command_rate_hz")
        )
        for name in names:
            if not values.get(name):
                answer = ask(f"{side}.{name} (blank = unknown): ").strip()
                if answer:
                    values[name] = float(answer) if name == "command_rate_hz" else answer
        show(f"Review {side}: {json.dumps(values, indent=2)}")
        payload[f"{side}_confirmed"] = ask(f"Confirm this {side} configuration? [y/N]: ").strip().lower() in {
            "y",
            "yes",
        }
    return ControllerProfile.from_dict(payload)


def resolve_controller(environment, *, profile=None, isaaclab_env=None, action_name=None, asset_name="robot"):
    """Return a frozen environment and discovery report without mutating user cfg."""
    from dataclasses import replace

    existing = environment.controller_profile
    if profile is not None and existing:
        supplied = profile.to_dict() if isinstance(profile, ControllerProfile) else profile
        if supplied != existing:
            raise ValueError("Conflicting controller profiles: update the environment or remove its old profile")
    supplied = profile if profile is not None else existing or None
    if supplied is not None and not isinstance(supplied, ControllerProfile):
        supplied = ControllerProfile.from_dict(supplied)
    if isaaclab_env is not None:
        detected = discover_controller(isaaclab_env, action_name=action_name, asset_name=asset_name)
        if supplied is not None:
            # Live config is authoritative; never let a JSON label hide an OSC
            # controller or let old confirmations survive a settings change.
            expected, actual = dict(supplied.simulation), dict(detected.simulation)
            expected.pop("frame", None)
            actual.pop("frame", None)
            # User-declared log boundary is not a discovered Isaac Lab setting.
            expected["settings"] = dict(expected.get("settings", {}))
            boundary = expected["settings"].pop("evidence_boundary", None)
            if expected != actual:
                raise ValueError(
                    "Live controller differs from the supplied profile; rediscover and confirm a new revision"
                )
            simulation = dict(detected.simulation)
            simulation["frame"] = supplied.simulation.get("frame", simulation.get("frame", ""))
            simulation["settings"] = dict(simulation["settings"])
            if boundary:
                simulation["settings"]["evidence_boundary"] = boundary
            supplied = replace(supplied, simulation=simulation)
        else:
            supplied = detected
    report = controller_report(
        supplied,
        legacy_confirmed=environment.controller_profile_confirmed,
        legacy_source=environment.controller_profile_source,
    )
    if supplied is not None:
        environment = replace(environment, controller_profile=supplied.to_dict())
    return environment, report
