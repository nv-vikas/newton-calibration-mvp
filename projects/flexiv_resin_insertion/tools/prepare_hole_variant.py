"""Verify the actual USD bore/chamfer and solve held-out poses above it."""

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
parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=0)
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
guess = np.array([-0.0347411846, 0.0339654728])
r = np.linalg.norm(vertices[:, :2] - guess, axis=1)
ring = vertices[(np.abs(vertices[:, 2] - 0.038) < 1e-7) & (np.abs(r - 0.012875) < 1e-5), :2]
assert len(ring) >= 100
fit = np.linalg.lstsq(np.column_stack([2 * ring, np.ones(len(ring))]), (ring * ring).sum(1), rcond=None)[0]
center = fit[:2]
radius = np.sqrt(fit[2] + (fit[:2] ** 2).sum())
top_ring = vertices[(np.abs(vertices[:, 2] - 0.04) < 1e-7) & (np.abs(r - 0.014875) < 1e-5)]
assert len(top_ring) >= 100 and abs(radius - 0.012875) < 1e-7
top_radius = float(np.linalg.norm(top_ring[:, :2] - center, axis=1).mean())
assert abs(top_radius - radius - 0.002) < 1e-7
all_holes = []
for cx, cy, diameter, chamfer in [
    (-0.0347411846, 0.0339654728, 0.02575, 0.002),
    (0.028202974, 0.033300563, 0.0255, 0.002),
    (-0.035849356, -0.030086866, 0.02525, 0.002),
    (0.030197681, -0.026983984, 0.02505, 0.002),
    (0.0, 0.0, 0.0255, 0.0),
]:
    radial = np.linalg.norm(vertices[:, :2] - [cx, cy], axis=1)
    bottom_ring = vertices[(np.abs(vertices[:, 2]) < 1e-7) & (np.abs(radial - diameter / 2) < 1e-5)]
    assert len(bottom_ring) >= 100
    all_holes.append(
        {"target_world_m": [0.15 + cx, cy, 0.79], "bore_radius_m": diameter / 2, "chamfer_height_m": chamfer}
    )
robot = Usd.Stage.Open(str(args.prepared / "Flexiv_Rizon4s_Grav.usd"))
fk, bounds = forward_model(robot)
base = np.array(original["scene_manifest"]["scene"]["base_world"])
names = [f"joint{i}" for i in range(1, 8)]
rows = []
rng = np.random.default_rng(932701)
for entry in original["starts"]:
    if entry["stage"] != args.stage or entry["split"] != "heldout":
        continue
    item = json.loads(json.dumps(entry))
    for key in ["gripper_pos", "peg_pos"]:
        item[key][0] += float(center[0])
        item[key][1] += float(center[1])
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
        seed = np.clip(np.asarray(item["joint_pos"]) + rng.normal(0, 0.25, 7), bounds[:, 0] + 0.04, bounds[:, 1] - 0.04)
        alternative = least_squares(residual, seed, bounds=(bounds[:, 0] + 0.03, bounds[:, 1] - 0.03), max_nfev=500)
        attempts += 1
        if np.linalg.norm(residual(alternative.x)) < np.linalg.norm(residual(solution.x)):
            solution = alternative
    error = float(np.linalg.norm(residual(solution.x)))
    if error > 1e-4:
        raise RuntimeError(f"Variant IK failed: {error}")
    item["joint_pos"] = solution.x.tolist()
    item["ik_error"] = error
    item["ik_attempts"] = attempts
    rows.append(item)
if not rows:
    raise RuntimeError(f"No held-out starts exist for stage {args.stage}")
result = {
    "label": "25.75",
    "fixture_moved": False,
    "target_world_m": [float(0.15 + center[0]), float(center[1]), 0.79],
    "target_in_fixture_m": [*center.tolist(), 0.04],
    "bore_radius_m": float(radius),
    "chamfer_height_m": 0.002,
    "chamfer_entry_radius_m": top_radius,
    "chamfer_angle_degrees": 45,
    "all_holes": all_holes,
    "basis": "USD collision-mesh vertices plus engraved label verified from source STL top section",
    "hole_usd_sha256": hashlib.sha256((args.prepared / "HoleBlock.usd").read_bytes()).hexdigest(),
    "prepared_sha256": hashlib.sha256((args.prepared / "prepared.json").read_bytes()).hexdigest(),
    "curriculum_stage": args.stage,
    "heldout_starts": rows,
    "scope": "new target experiment; original checkpoint qualification does not validate this target",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "heldout_starts"}, indent=2))
print(f"Solved {len(rows)} held-out poses; max IK error {max(r['ik_error'] for r in rows):.3g}")
