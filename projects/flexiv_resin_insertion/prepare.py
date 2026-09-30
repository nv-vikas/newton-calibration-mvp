"""Create separate resin USDs and above-hole IK banks; no hardware I/O."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def matrix(pos, quat):
    result = np.eye(4)
    result[:3, 3] = pos
    result[:3, :3] = Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix()
    return result


def forward_model(stage):
    """Read kinematics from the supplied USD, not a hand-written Flexiv model."""
    root = str(stage.GetDefaultPrim().GetPath())
    joints = []
    bounds = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parent or not child:
            continue
        joints.append(
            (
                prim.GetName(),
                str(parent[0]),
                str(child[0]),
                matrix(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                np.linalg.inv(matrix(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())),
                "XYZ".index(prim.GetAttribute("physics:axis").Get() or "Z"),
            )
        )
        if prim.GetName() in [f"joint{i}" for i in range(1, 8)]:
            bounds[prim.GetName()] = np.deg2rad(
                [prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()]
            )

    def fk(q):
        poses = {root + "/base_link": np.eye(4)}
        pending = list(joints)
        while pending:
            count = len(pending)
            for item in pending[:]:
                name, parent, child, a, b, axis = item
                if parent not in poses:
                    continue
                motion = np.eye(4)
                rot = np.zeros(3)
                rot[axis] = q.get(name, 0.0)
                motion[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
                poses[child] = poses[parent] @ a @ motion @ b
                pending.remove(item)
            if count == len(pending):
                raise ValueError("Disconnected USD kinematics")
        return poses[root + "/Grav_gripper/gripper_base"]

    return fk, np.array([bounds[f"joint{i}"] for i in range(1, 8)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(__file__).with_name("experiment.json").read_text())
    manifest = json.loads((source / "asset_manifest.json").read_text())
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.usd")}
    for name, old_density in [("Peg_25", 7850.0), ("HoleBlock", 2700.0), ("Flexiv_Rizon4s_Grav", None)]:
        dest = output / f"{name}.usd"
        if dest.exists():
            raise FileExistsError(f"Refusing to overwrite prepared asset: {dest}")
        stage = Usd.Stage.CreateNew(str(dest))
        stage.GetRootLayer().TransferContent(Usd.Stage.Open(str(source / f"{name}.usd")).Flatten())
        for prim in list(stage.Traverse()):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in list(stage.Traverse()):
            if old_density:
                mass = UsdPhysics.MassAPI(prim)
                if mass and mass.GetMassAttr().HasAuthoredValueOpinion():
                    ratio = config["resin"]["density_kg_m3"] / old_density
                    mass.GetMassAttr().Set(mass.GetMassAttr().Get() * ratio)
                    mass.CreateDensityAttr(config["resin"]["density_kg_m3"])
                    inertia = mass.GetDiagonalInertiaAttr().Get()
                    if inertia is not None:
                        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[v * ratio for v in inertia]))
                if prim.HasAPI(UsdPhysics.MaterialAPI):
                    material = UsdPhysics.MaterialAPI(prim)
                    material.CreateStaticFrictionAttr(config["resin"]["static_friction"])
                    material.CreateDynamicFrictionAttr(config["resin"]["dynamic_friction"])
                    material.CreateRestitutionAttr(0.0)
                if prim.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(prim)
                    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(*((0.15, 0.6, 0.85) if name == "Peg_25" else (0.8, 0.65, 0.3)))
                    )
            elif "/collisions/" in str(prim.GetPath()) and prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        stage.GetDefaultPrim().SetCustomDataByKey("experiment:calibrated", False)
        if old_density:
            stage.GetDefaultPrim().SetCustomDataByKey("experiment:materialBasis", config["resin"]["basis"])
        stage.GetRootLayer().Save()

    robot_stage = Usd.Stage.Open(str(output / "Flexiv_Rizon4s_Grav.usd"))
    fk, limits = forward_model(robot_stage)
    names = [f"joint{i}" for i in range(1, 8)]
    home = np.array([manifest["scene"]["joint_positions_rad"][n] for n in names])
    base = np.array(manifest["scene"]["base_world"])
    peg_offset = np.array(manifest["scene"]["peg_in_gripper"])
    peg_offset[2, 3] += config["geometry"]["peg_extension_from_original_grasp_m"]
    scene_manifest = json.loads(json.dumps(manifest))
    scene_manifest["scene"]["peg_in_gripper"] = peg_offset.tolist()
    cache = UsdGeom.XformCache()
    base_prim = robot_stage.GetPrimAtPath("/Rizon4s/Grav_gripper/gripper_base")
    gripper_world = np.array(cache.GetLocalToWorldTransform(base_prim)).T
    exposed = []
    for side in ["left", "right"]:
        prim = robot_stage.GetPrimAtPath(f"/Rizon4s/Grav_gripper/{side}_finger_tip/collisions")
        vertices = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        transform = np.linalg.inv(gripper_world) @ np.array(cache.GetLocalToWorldTransform(prim)).T
        local = (transform @ np.column_stack([vertices, np.ones(len(vertices))]).T).T[:, :3]
        exposed.append(float(peg_offset[2, 3] + config["geometry"]["peg_length_m"] - local[:, 2].max()))
    if min(exposed) < config["geometry"]["success_depth_m"] + 0.005:
        raise ValueError("Grasp leaves insufficient exposed peg length for insertion plus fingertip clearance")
    rng = np.random.default_rng(config["training"]["seed"])
    records = []
    # All resets start ABOVE the hole; none starts already inserted.
    for stage_index, (xy_range, tilt_range) in enumerate([(0.00005, 0.0), (0.001, 0.01), (0.004, 0.035)]):
        for index in range(64):
            peg_target = np.eye(4)
            peg_target[:3, :3] = Rotation.from_euler(
                "xy", [np.pi + rng.uniform(-tilt_range, tilt_range), rng.uniform(-tilt_range, tilt_range)]
            ).as_matrix()
            peg_target[:3, 3] = [
                0.15 + rng.uniform(-xy_range, xy_range),
                rng.uniform(-xy_range, xy_range),
                0.04 + 0.75 + 0.075 + rng.uniform(0.003, 0.012),
            ]
            target = np.linalg.inv(base) @ peg_target @ np.linalg.inv(peg_offset)

            def residual(q, target=target):
                pose = fk(dict(zip(names, q)))
                return np.r_[
                    (pose[:3, 3] - target[:3, 3]) * 10,
                    Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec(),
                ]

            solution = least_squares(residual, home, bounds=(limits[:, 0] + 0.03, limits[:, 1] - 0.03), max_nfev=400)
            for _attempt in range(8):
                if np.linalg.norm(residual(solution.x)) <= 1e-4:
                    break
                seed = np.clip(home + rng.normal(0.0, 0.25, 7), limits[:, 0] + 0.04, limits[:, 1] - 0.04)
                alternative = least_squares(
                    residual, seed, bounds=(limits[:, 0] + 0.03, limits[:, 1] - 0.03), max_nfev=400
                )
                if np.linalg.norm(residual(alternative.x)) < np.linalg.norm(residual(solution.x)):
                    solution = alternative
            error = float(np.linalg.norm(residual(solution.x)))
            if error > 1e-4:
                raise RuntimeError(f"IK failed for stage {stage_index}, sample {index}: {error}")
            records.append(
                {
                    "stage": stage_index,
                    "joint_pos": solution.x.tolist(),
                    "peg_pos": peg_target[:3, 3].tolist(),
                    "peg_quat_xyzw": Rotation.from_matrix(peg_target[:3, :3]).as_quat().tolist(),
                    "gripper_pos": (base @ target)[:3, 3].tolist(),
                    "gripper_quat_xyzw": Rotation.from_matrix((base @ target)[:3, :3]).as_quat().tolist(),
                    "ik_error": error,
                    "split": "heldout" if index >= 48 else "train",
                }
            )
    result = {
        "experiment": config,
        "source_asset_sha256": hashes,
        "source_manifest": manifest,
        "scene_manifest": scene_manifest,
        "grasp_readiness": {
            "peg_extension_from_original_m": config["geometry"]["peg_extension_from_original_grasp_m"],
            "exposed_peg_length_m_by_finger": exposed,
            "required_depth_m": config["geometry"]["success_depth_m"],
            "minimum_fingertip_clearance_m": 0.005,
            "basis": "authored joint geometry; runtime contact qualification still required",
        },
        "starts": records,
        "qualification": "kinematic starts only; requires physical grasp/contact qualification",
    }
    (output / "prepared.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prepared": len(records), "output": str(output), "calibrated": False}), flush=True)


if __name__ == "__main__":
    main()
