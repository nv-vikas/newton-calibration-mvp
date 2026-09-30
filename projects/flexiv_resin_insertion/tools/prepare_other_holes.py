"""Verify the remaining holes and prepare matched resets; never edits USD assets."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--indices", type=int, nargs="+", default=[14, 7, 2, 12],
                    help="Explicit source-bank indices; use unused starts for qualification holdouts")
parser.add_argument("--holes", nargs="+", help="Optional exact hole keys to prepare")
args = parser.parse_args()
sys.path.insert(0, "/work/flexiv_resin_insertion")
from prepare import forward_model

if args.output.exists():
    raise FileExistsError(args.output)
original = json.loads((args.prepared / "prepared.json").read_text())
usd = Usd.Stage.Open(str(args.prepared / "HoleBlock.usd"))
collider = usd.GetPrimAtPath("/HoleBlock/Collision")
assert UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() == "none"
vertices = np.asarray(UsdGeom.Mesh(collider).GetPointsAttr().Get(), dtype=float)
catalog = [
    ("center_2550", "25.50 mm center", 0.0, 0.0, 0.0255, 0.0),
    ("chamfer_2550", "25.50 mm chamfered", 0.028202974, 0.033300563, 0.0255, 0.002),
    ("chamfer_2525", "25.25 mm chamfered", -0.035849356, -0.030086866, 0.02525, 0.002),
    ("chamfer_2505", "25.05 mm chamfered", 0.030197681, -0.026983984, 0.02505, 0.002),
    ("chamfer_2575", "25.75 mm chamfered", -0.0347411846, 0.0339654728, 0.02575, 0.002),
]
geometry = []
for key, label, cx, cy, diameter, chamfer in catalog:
    radial = np.linalg.norm(vertices[:, :2] - [cx, cy], axis=1)
    ring = vertices[(np.abs(vertices[:, 2]) < 1e-7) & (np.abs(radial - diameter / 2) < 1e-5), :2]
    assert len(ring) >= 100, (key, "bottom vertices", len(ring))
    fit = np.linalg.lstsq(np.column_stack([2 * ring, np.ones(len(ring))]), (ring * ring).sum(1), rcond=None)[0]
    center = fit[:2]
    radius = float(np.sqrt(fit[2] + (center * center).sum()))
    assert abs(radius - diameter / 2) < 1e-7
    top = vertices[(np.abs(vertices[:, 2] - 0.04) < 1e-7) & (np.abs(radial - diameter / 2 - chamfer) < 1e-5)]
    assert len(top) >= 100, (key, "mouth vertices", len(top))
    mouth_radius = float(np.linalg.norm(top[:, :2] - center, axis=1).mean())
    assert abs(mouth_radius - radius - chamfer) < 1e-7
    if chamfer:
        throat = vertices[(np.abs(vertices[:, 2] - 0.038) < 1e-7) & (np.abs(radial - diameter / 2) < 1e-5)]
        assert len(throat) >= 100, (key, "chamfer throat", len(throat))
    geometry.append(
        {
            "key": key,
            "label": label,
            "target_world_m": [float(0.15 + center[0]), float(center[1]), 0.79],
            "target_in_fixture_m": [float(center[0]), float(center[1]), 0.04],
            "bore_radius_m": radius,
            "nominal_diameter_mm": diameter * 1000,
            "radial_clearance_m": radius - 0.0125,
            "chamfer_height_m": chamfer,
            "mouth_radius_m": mouth_radius,
        }
    )

robot = Usd.Stage.Open(str(args.prepared / "Flexiv_Rizon4s_Grav.usd"))
fk, bounds = forward_model(robot)
base = np.array(original["scene_manifest"]["scene"]["base_world"])
names = [f"joint{i}" for i in range(1, 8)]
heldout = [r for r in original["starts"] if r["stage"] == 1 and r["split"] == "heldout"]
assert len(heldout) == 16
indices = args.indices
if len(indices) != len(set(indices)) or any(i < 0 or i >= len(heldout) for i in indices):
    raise ValueError("Require distinct source-bank indices in range")
rng = np.random.default_rng(932701)
jobs = []
chosen_holes = [h for h in geometry if h["key"] in args.holes] if args.holes else geometry[:4]
if args.holes and set(args.holes) != {h["key"] for h in chosen_holes}:
    raise ValueError("Unknown hole key")
for hole in chosen_holes:
    starts = []
    for index in indices:
        item = json.loads(json.dumps(heldout[index]))
        for key in ["gripper_pos", "peg_pos"]:
            item[key][0] += hole["target_in_fixture_m"][0]
            item[key][1] += hole["target_in_fixture_m"][1]
        desired = np.eye(4)
        desired[:3, :3] = Rotation.from_quat(item["gripper_quat_xyzw"]).as_matrix()
        desired[:3, 3] = item["gripper_pos"]
        desired = np.linalg.inv(base) @ desired

        def residual(q, desired=desired):
            pose = fk(dict(zip(names, q, strict=True)))
            return np.r_[
                (pose[:3, 3] - desired[:3, 3]) * 10, Rotation.from_matrix(desired[:3, :3].T @ pose[:3, :3]).as_rotvec()
            ]

        solution = least_squares(
            residual, item["joint_pos"], bounds=(bounds[:, 0] + 0.03, bounds[:, 1] - 0.03), max_nfev=500
        )
        attempts = 1
        for _ in range(8):
            if np.linalg.norm(residual(solution.x)) <= 1e-4:
                break
            seed = np.clip(
                np.array(item["joint_pos"]) + rng.normal(0, 0.25, 7), bounds[:, 0] + 0.04, bounds[:, 1] - 0.04
            )
            alternative = least_squares(residual, seed, bounds=(bounds[:, 0] + 0.03, bounds[:, 1] - 0.03), max_nfev=500)
            attempts += 1
            if np.linalg.norm(residual(alternative.x)) < np.linalg.norm(residual(solution.x)):
                solution = alternative
        error = float(np.linalg.norm(residual(solution.x)))
        if error > 1e-4:
            raise RuntimeError(f"IK failed: {hole['key']}, heldout index {index}, error {error}")
        item.update(joint_pos=solution.x.tolist(), ik_error=error, ik_attempts=attempts, source_heldout_index=index)
        starts.append(item)
    jobs.append({**hole, "starts": starts, "recorded_start_index": 0})
    print(json.dumps({"hole": hole["key"], "max_ik_error": max(r["ik_error"] for r in starts)}), flush=True)

result = {
    "schema": "frozen-policy-other-holes/v1",
    "curriculum_stage": 1,
    "jobs": jobs,
    "all_holes": geometry,
    "common_source_heldout_indices": indices,
    "checkpoint_name": "model_731.pt",
    "training_updates": 732,
    "checkpoint_sha256": "b2ccab2b5e03a4a61371c94021d06ff0408f607e59fbef6977ee76085c578a90",
    "hole_usd_sha256": hashlib.sha256((args.prepared / "HoleBlock.usd").read_bytes()).hexdigest(),
    "prepared_sha256": hashlib.sha256((args.prepared / "prepared.json").read_bytes()).hexdigest(),
    "fixture_moved": False,
    "physics_parameters_changed": False,
    "policy_retrained": False,
    "completion_criteria": "Original 35 mm depth, 0.35 mm radial, 1 degree tilt, 0.25 s hold, 0.2 mm sampled penetration gate",
    "qualification": "Report free-peg bore/rim probes separately. Failed probes qualify interpretation, not hidden policy retries.",
    "precision_caveat": "The 25.05 mm hole has 25 um radial clearance, below the unchanged 50 um shape margin/gap and 200 um invalidity gate; nominal task completion is not a precision-fit certificate.",
    "scope": "Four matched simulated starts per hole; locations and joint configurations also differ. No isolated diameter causality or real-transfer claim.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "holes": len(jobs), "episodes": len(jobs) * len(indices)}), flush=True)
