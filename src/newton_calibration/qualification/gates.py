"""Acceptance belongs to the toolkit, not the proposing agent or optimizer."""

from __future__ import annotations

from .contracts import Gates, positive


def geometry_gate(data, gates: Gates):
    clearance = data["minimum_radial_clearance_m"]
    positive(clearance, "measured clearance")
    positive(data["geometry_error_bound_m"], "geometry error", zero=True)
    reasons = []
    if data["coverage"] != "full_insertion_depth":
        reasons.append("geometry inspection does not cover the full inserted section")
    if data["geometry_error_bound_m"] > min(gates.max_geometry_error_m, clearance / 4):
        reasons.append("geometry measurement/representation error is too large for this clearance")
    if gates.max_wall_penetration_m >= clearance / 2:
        reasons.append("wall penetration gate is too loose for the measured clearance; create a stricter recipe")
    if data.get("collision_enabled") is not True or data.get("bore_preserved") is not True:
        reasons.append("collision setup does not preserve a solid fixture with an open bore")
    return {"passed": not reasons, "reasons": reasons}


def validate_rows(data, expected_ids, *, policy=False):
    rows = data["trials"]
    if sorted(r["id"] for r in rows) != sorted(expected_ids):
        raise ValueError("Missing, duplicated or substituted trials")
    for row in rows:
        for name in ("max_wall_penetration_m", "max_grasp_displacement_m", "hold_s"):
            positive(row[name], name, zero=True)
        # Depth can be negative before entry; finite, signed value is legitimate.
        positive(abs(row["depth_m"]), "depth", zero=True)
        if type(row["valid"]) is not bool:
            raise ValueError("Trial valid must be a boolean")
        if policy and type(row["policy_success"]) is not bool:
            raise ValueError("Policy completion flag must be boolean")
    return rows


def probes_gate(data, ids, gates: Gates):
    rows = validate_rows(data, ids)
    reasons = []
    for row in rows:
        positive(abs(row["rim_depth_m"]), "rim depth", zero=True)
        if not row["valid"] or not gates.insertion_depth_m <= row["depth_m"] <= gates.maximum_depth_m:
            reasons.append(f"{row['id']}: centered-bore probe did not seat validly")
        if row["rim_depth_m"] > gates.rim_max_depth_m:
            reasons.append(f"{row['id']}: solid-rim probe admitted the peg")
        if row["max_wall_penetration_m"] > gates.max_wall_penetration_m:
            reasons.append(f"{row['id']}: excessive wall penetration")
    return {"passed": not reasons, "reasons": reasons}


def compare_probes(left, right, ids, tolerance):
    a = {r["id"]: r for r in validate_rows(left, ids)}
    b = {r["id"]: r for r in validate_rows(right, ids)}
    error = max(abs(a[i][key] - b[i][key]) for i in ids for key in ("depth_m", "rim_depth_m"))
    return {"passed": error <= tolerance, "max_difference_m": error, "tolerance_m": tolerance}


def insertion_gate(data, ids, gates: Gates, *, policy=False):
    rows = validate_rows(data, ids, policy=policy)
    checks = {}
    for row in rows:
        checks[row["id"]] = bool(
            row["valid"]
            and gates.insertion_depth_m <= row["depth_m"] <= gates.maximum_depth_m
            and row["hold_s"] >= gates.hold_s
            and row["max_wall_penetration_m"] <= gates.max_wall_penetration_m
            and row["max_grasp_displacement_m"] <= gates.max_grasp_displacement_m
            and (not policy or row["policy_success"])
        )
    successes = sum(checks.values())
    return {
        "passed": successes / len(rows) >= (gates.minimum_policy_success_fraction if policy else 1),
        "successes": successes,
        "trials": len(rows),
        "per_trial": checks,
        "scope": "simulated episodes, not independent real trials or a statistical confidence guarantee",
    }


def controlled_gate(data, ids, gates: Gates):
    result = insertion_gate(data, ids, gates)
    checks = {}
    for name in ("hold", "free_motion"):
        phase = data["diagnostic_phases"][name]
        for metric in ("max_wall_penetration_m", "max_grasp_displacement_m"):
            positive(phase[metric], metric, zero=True)
        checks[name] = (
            sorted(phase["completed_trial_ids"]) == sorted(ids)
            and phase["max_wall_penetration_m"] <= gates.max_wall_penetration_m
            and phase["max_grasp_displacement_m"] <= gates.max_grasp_displacement_m
        )
    result["diagnostic_phases"] = checks
    result["passed"] = result["passed"] and all(checks.values())
    return result
