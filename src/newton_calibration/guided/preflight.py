"""Offline, source-backed assistance. Proposals never grant permission to fit.

These tools inspect supplied configuration or run a conditional command model.
They do not connect to a robot, relabel reconstructed commands as measurements,
confirm hardware facts, or modify a guided session.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from newton_calibration.core.io import sha256_file

BASELINE_FIELDS = (
    "base_stiffness_by_joint",
    "base_damping_by_joint",
    "base_effort_limit_by_joint",
)


def inspect_simulation_profile(path, joint_map):
    """Extract explicit simulation priors; never substitute hardware metadata.

    A complete, matching map and finite per-joint values are required. Partial
    guesses and generic scalar fallbacks cannot silently complete the proposal.
    Values use the toolkit's runtime units (rad, N m), not raw USD drive units.
    """
    source = Path(path).expanduser().resolve()
    before = sha256_file(source)
    data = json.loads(source.read_text())
    if not joint_map or any(not isinstance(k, str) or not isinstance(v, str) for k, v in joint_map.items()):
        raise ValueError("Provide a nonempty logical-to-USD joint map")
    if len(set(joint_map.values())) != len(joint_map):
        raise ValueError("Joint map contains duplicate target joints")
    if data.get("joint_map") != joint_map:
        raise ValueError("Simulation source joint map does not match this run")
    values = {}
    for field in BASELINE_FIELDS:
        entries = data.get(field)
        if not isinstance(entries, dict) or set(entries) != set(joint_map):
            raise ValueError(f"{field} must explicitly cover every selected joint")
        for value in entries.values():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} contains an invalid value")
            if field == "base_effort_limit_by_joint" and value == 0:
                raise ValueError("Effort limits must be positive")
        values[field] = dict(entries)
    if before != sha256_file(source):
        raise ValueError("Simulation source changed during inspection")
    return {
        "schema": "newton.simulation-profile-proposal/v1",
        "source": {"path": str(source), "sha256": before, "kind": "simulation_prior"},
        "environment_patch": values,
        "complete": True,
        "hardware_confirmed": False,
        "runtime_validated": False,
        "notice": "Explicit simulation starting values only; not measured hardware gains or approved hardware limits.",
    }


def replay_slew_limiter(times, requested, *, velocity_limit, minimum_dt=0.001, initial_target=None, initial_time=None):
    """Pure conditional model of a scalar velocity-limited target stream.

    Matches the supplied Flexiv shim's recurrence, not the downstream firmware.
    With no initial state the first target passes unchanged. That is a declared
    model assumption, NOT evidence that a historical stream began that way.
    Call at command event times, not at every physics tick.
    """
    times = np.asarray(times, dtype=float)
    requested = np.asarray(requested, dtype=float)
    if times.ndim != 1 or not len(times) or requested.ndim != 2 or requested.shape[0] != len(times):
        raise ValueError("Need nonempty event times and an N-by-joints target array")
    if not requested.shape[1] or not np.isfinite(times).all() or not np.isfinite(requested).all():
        raise ValueError("Times and joint targets must be finite")
    if not (np.diff(times) > 0).all():
        raise ValueError("Command times must be strictly increasing")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (velocity_limit, minimum_dt)):
        raise ValueError("Velocity limit and minimum dt must be finite and positive")
    if (initial_target is None) != (initial_time is None):
        raise ValueError("Initial target and time must be supplied together")
    previous = None
    previous_time = None
    if initial_target is not None:
        previous = np.asarray(initial_target, dtype=float).copy()
        if previous.shape != requested.shape[1:] or not np.isfinite(previous).all():
            raise ValueError("Initial target must cover every joint with finite values")
        if not math.isfinite(initial_time) or initial_time > times[0]:
            raise ValueError("Initial time must be finite and precede the first event")
        previous_time = float(initial_time)
    output = np.empty_like(requested)
    for index, (time, target) in enumerate(zip(times, requested)):
        if previous is None:
            current = target.copy()
        else:
            limit = velocity_limit * max(float(time - previous_time), minimum_dt)
            current = previous + np.clip(target - previous, -limit, limit)
        output[index] = current
        previous, previous_time = current, float(time)
    return output


def inspect_usd_properties(path):
    """Inventory authored mechanics in OpenUSD, without initializing physics.

    No mass/inertia inference, angular-drive unit conversion or asset repair is
    performed. Missing/default properties are reported, not manufactured.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    source = Path(path).expanduser().resolve()
    before = sha256_file(source)
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError("Cannot open USD")

    def value(attribute):
        if not attribute or not attribute.HasAuthoredValueOpinion():
            return None
        raw = attribute.Get()
        if isinstance(raw, (str, bool, int, float)) or raw is None:
            return raw
        return list(raw)

    bodies, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(
                {
                    "path": str(prim.GetPath()),
                    "mass": value(prim.GetAttribute("physics:mass")),
                    "center_of_mass": value(prim.GetAttribute("physics:centerOfMass")),
                    "diagonal_inertia": value(prim.GetAttribute("physics:diagonalInertia")),
                }
            )
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints.append(
                {
                    "name": prim.GetName(),
                    "path": str(prim.GetPath()),
                    "axis": value(prim.GetAttribute("physics:axis")),
                    "lower_limit_deg": value(prim.GetAttribute("physics:lowerLimit")),
                    "upper_limit_deg": value(prim.GetAttribute("physics:upperLimit")),
                    "authored_angular_drive": {
                        key: value(prim.GetAttribute(f"drive:angular:physics:{key}"))
                        for key in ("type", "stiffness", "damping", "maxForce")
                    },
                }
            )
    if before != sha256_file(source):
        raise ValueError("USD changed during inspection")
    return {
        "schema": "newton.authored-mechanics-audit/v1",
        "source": {"path": str(source), "sha256": before},
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "kilograms_per_unit": UsdPhysics.GetStageKilogramsPerUnit(stage),
        "bodies": bodies,
        "joints": joints,
        "hardware_match_confirmed": False,
        "physics_initialized": False,
        "notice": "Authored properties, not runtime readback. None/zero inertia needs importer review, not invented values.",
    }
