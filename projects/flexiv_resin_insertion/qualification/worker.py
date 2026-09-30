"""Actual Isaac Lab/Newton reference worker. No optimizer, training or hardware imports.

Each experiment gets a fresh process and environment. Frozen source assets and
checkpoint are read-only inputs. The core determines acceptance from measurements.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from newton_calibration.core.io import atomic_write_json, sha256_file
from newton_calibration.qualification.contracts import Settings, digest
from newton_calibration.qualification.geometry import inspect_bore_triangles


def read(path):
    return json.loads(Path(path).read_text())


def input_files(binding_path, binding):
    source, prepared = Path(binding["source"]), Path(binding["prepared"])
    files = {
        "asset": prepared / "Peg_25.usd",
        "fixture": prepared / "HoleBlock.usd",
        "robot": prepared / "Flexiv_Rizon4s_Grav.usd",
        "prepared": prepared / "prepared.json",
        "policy": Path(binding["checkpoint"]),
        "controller": source / "task.py",
        "scene": source / "experiment.json",
        "trial_bank": Path(binding["spec"]),
        "ppo": Path(binding["ppo_config"]),
        "binding": Path(binding_path),
    }
    for prefix, directory in (
        ("task", source),
        ("worker", Path(__file__).parent),
        ("tool", Path(binding["tools"])),
    ):
        for path in sorted(directory.glob("*.py")):
            files[f"{prefix}_{path.name}"] = path
    import newton_calibration

    core = Path(newton_calibration.__file__).parent
    for path in sorted(core.rglob("*.py")):
        files[f"core_{path.relative_to(core)}"] = path
    return {key: str(path.resolve()) for key, path in files.items()}


def selected_job(binding):
    spec = read(binding["spec"])
    jobs = [job for job in spec["jobs"] if job["key"] == binding["hole"]]
    if len(jobs) != 1:
        raise ValueError("Binding must select exactly one prepared target hole")
    return spec, jobs[0]


def describe(binding_path, binding):
    files = input_files(binding_path, binding)
    _, job = selected_job(binding)
    identities = {}
    for row in job["starts"]:
        # Use actual initial conditions, not the role or assigned ID, to detect split leakage.
        trial_id = str(row["source_heldout_index"])
        if trial_id in identities:
            raise ValueError("Duplicate reference-bank ID")
        identities[trial_id] = digest({k: row[k] for k in ("joint_pos", "peg_pos", "peg_quat_xyzw")})
    return {
        "runtime": "isaaclab_newton",
        "source": "simulation",
        "identity": {
            "adapter": "flexiv-resin-precision-insertion/v1",
            "newton": importlib.metadata.version("newton"),
            "isaaclab": importlib.metadata.version("isaaclab"),
            "worker": sha256_file(__file__),
            "controller": sha256_file(files["controller"]),
            "target": binding["hole"],
            "physics": "Newton contacts + MuJoCo Warp",
            "hardware_supported": False,
        },
        "input_files": sorted(set(files.values())),
        "trial_fingerprints": identities,
        "capabilities": ["geometry", "probes", "controlled_insertion", "frozen_policy"],
        "parameter_bounds": {
            "dt_s": [1 / 15360, 1 / 240],
            "margin_m": [0, 50e-6],
            "gap_m": [0, 100e-6],
            "iterations": [50, 1000],
            # Pinned mujoco_warp/_src/io.py put_model clamps tolerance to 1e-6.
            # Expose the real supported floor instead of bypassing the runtime.
            "tolerance": [1e-6, 1e-5],
        },
        "limits": "Known rigid cylindrical peg/straight faceted bore; no arbitrary geometry or real-force calibration",
    }


def mesh(stage_path, prim_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(stage_path))
    prim = stage.GetPrimAtPath(prim_path)
    geometry = UsdGeom.Mesh(prim)
    if not geometry or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Expected explicit collision mesh")
    points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
    if not np.allclose(transform, np.eye(4), atol=1e-12):
        raise ValueError("Reference collider transform changed; inspect/rebind rather than silently use local points")
    counts = np.asarray(geometry.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError("Reference inspector requires triangles")
    indices = np.asarray(geometry.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    return (
        points,
        points[indices],
        bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()),
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
    )


def inspect_geometry(binding):
    _, job = selected_job(binding)
    prepared = Path(binding["prepared"])
    peg_points, _, peg_collision, peg_approx = mesh(prepared / "Peg_25.usd", "/Peg_25/Collision")
    _, triangles, block_collision, block_approx = mesh(prepared / "HoleBlock.usd", "/HoleBlock/Collision")
    if peg_approx != "convexHull" or block_approx != "none":
        raise ValueError("Pinned collision representations changed")
    peg_radius = float(np.linalg.norm(peg_points[:, :2], axis=1).max())
    if (
        abs(peg_radius - 0.0125) > 1e-7
        or abs(peg_points[:, 2].min()) > 1e-7
        or abs(peg_points[:, 2].max() - 0.075) > 1e-7
    ):
        raise ValueError("This scene adapter requires the supplied 25 mm x 75 mm peg")
    result = inspect_bore_triangles(
        triangles,
        center_xy=np.array(job["target_in_fixture_m"][:2]),
        radius_m=job["bore_radius_m"],
        height_m=0.04,
        chamfer_m=job["chamfer_height_m"],
        peg_radius_m=peg_radius,
    )
    return {
        **result,
        "collision_enabled": peg_collision and block_collision,
        "bore_preserved": block_approx == "none",
        "peg_radius_m": peg_radius,
        "hole": binding["hole"],
    }


def physics(binding, request, output_dir, publish):
    # Isaac Lab must launch before importing the environment or physics backend.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    launch_args = parser.parse_args([])
    launch_args.backend, launch_args.headless = "newton", True
    launch_args.enable_cameras = request["video"]
    launch_args.device = binding.get("device", "cuda:0")
    launcher = AppLauncher(launch_args)
    env = None
    writers = []
    try:
        # Never import USD in this process before AppLauncher: the standalone
        # USD wheel and Kit's USD runtime can otherwise collide during startup.
        geometry = inspect_geometry(binding)
        import torch
        from isaaclab.physics import PhysicsEvent
        from isaaclab.utils.math import compute_pose_error, quat_apply, quat_conjugate, quat_mul
        from isaaclab_newton.physics.newton_manager import NewtonManager
        from newton import ShapeFlags

        sys.path.insert(0, binding["source"])
        sys.path.insert(0, binding["tools"])
        from other_hole_env import environment_class
        from task import ResinPegCfg, tensor

        original_spec, original_job = selected_job(binding)
        bank = {str(row["source_heldout_index"]): row for row in original_job["starts"]}
        selected = [bank[i] for i in request["trials"]]
        spec = {**original_spec, "jobs": [{**original_job, "starts": selected, "recorded_start_index": 0}]}
        atomic_write_json(output_dir / "experiment_spec.json", spec)
        base_env = environment_class(output_dir / "experiment_spec.json", capture=request["video"])
        gates = request["gates"]
        inner_radius, peg_radius = geometry["inscribed_bore_radius_m"], geometry["peg_radius_m"]
        chamfer = original_job["chamfer_height_m"]
        shape_binding = {}

        class PrecisionEnv(base_env):
            def _setup_scene(self):
                super()._setup_scene()
                # Bind the two task-object collider families before model finalization,
                # not through import defaults that authored USD attributes may override.
                self.qualification_binding_handle = NewtonManager.register_callback(
                    configure_builder,
                    PhysicsEvent.MODEL_INIT,
                    order=100000,
                    name="qualification_object_contact_settings",
                    wrap_weak_ref=False,
                )
                self.qualification_solver_handle = NewtonManager.register_callback(
                    configure_solver_attributes,
                    PhysicsEvent.PHYSICS_READY,
                    order=100000,
                    name="qualification_solver_model_attributes",
                    wrap_weak_ref=False,
                )

            def metrics(self):
                values = super().metrics()
                pose = tensor(self.peg.data.root_pose_w)
                root = pose[:, :3] - self.scene.env_origins
                axis = quat_apply(
                    pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
                )
                tip = root + 0.075 * axis
                # Enclose the convex peg in its measured cylinder. For a tilted
                # axis the horizontal cross-section fits inside radius r/|az|.
                # This conservative full-axis envelope replaces sparse surface sampling.
                az = axis[:, 2].abs().clamp_min(1e-4)
                radial_envelope = peg_radius / az
                cap_extension = peg_radius * torch.sqrt((1 - axis[:, 2] ** 2).clamp_min(0))
                lower = torch.minimum(root[:, 2], tip[:, 2]) - cap_extension
                upper = torch.maximum(root[:, 2], tip[:, 2]) + cap_extension
                wall = torch.zeros(self.num_envs, device=self.device)
                target = self.targets[:, :2]
                for bottom, top, slope in ((0.75, 0.79 - chamfer, 0.0), (0.79 - chamfer, 0.79, 1.0)):
                    if top <= bottom:
                        continue
                    lo, hi = lower.clamp_min(bottom), upper.clamp_max(top)
                    excess = torch.zeros_like(wall)
                    for z in (lo, hi):
                        axis_z = torch.where(axis[:, 2].abs() < 1e-4, -torch.ones_like(az) * 1e-4, axis[:, 2])
                        center = root[:, :2] + ((z - root[:, 2]) / axis_z)[:, None] * axis[:, :2]
                        available = inner_radius + slope * (z - bottom)
                        excess = torch.maximum(excess, (center - target).norm(dim=-1) + radial_envelope - available)
                    # Entry overlap is bounded by distance below the fixture top.
                    # This is a conservative screen, not a contact-force estimate.
                    wall = torch.maximum(
                        wall, torch.where(lo <= hi, torch.minimum(excess, (0.79 - lower).clamp_min(0)), 0)
                    )
                values["wall_violation"] = wall
                values["invalid"] |= (wall > gates["max_wall_penetration_m"]) | (
                    values["slip"] > gates["max_grasp_displacement_m"]
                )
                values["seated"] = (
                    (values["depth"] >= gates["insertion_depth_m"])
                    & (values["depth"] <= gates["maximum_depth_m"])
                    & ~values["invalid"]
                )
                if hasattr(self, "qualification_wall"):
                    self.qualification_wall = torch.maximum(self.qualification_wall, wall)
                    self.qualification_slip = torch.maximum(self.qualification_slip, values["slip"])
                return values

            def _apply_action(self):
                super()._apply_action()
                self.metrics()  # capture conservative geometry/grasp maxima at physics cadence

        settings = Settings(**request["settings"])
        settings.validate_rate(request["policy_hz"])
        cfg = ResinPegCfg()
        cfg.seed = 952705
        cfg.prepared_dir = binding["prepared"]
        cfg.scene.num_envs = len(selected)
        cfg.sim.device = launch_args.device
        cfg.evaluation = True
        cfg.curriculum_stage = original_spec["curriculum_stage"]
        cfg.sim.dt = settings.dt_s
        cfg.decimation = round(1 / (settings.dt_s * request["policy_hz"]))
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.physics.solver_cfg.iterations = settings.iterations
        cfg.sim.physics.solver_cfg.tolerance = settings.tolerance

        def configure_builder(*_args, **_kwargs):
            builder = NewtonManager._builder
            labels = list(builder.shape_label)
            ids = [
                i
                for i, label in enumerate(labels)
                if ("/Peg/" in label or "/HoleBlock/" in label)
                and int(builder.shape_flags[i]) & int(ShapeFlags.COLLIDE_SHAPES)
            ]
            if len(ids) != 2 * len(selected):
                raise ValueError(f"Expected exactly peg + fixture colliders per environment, found {len(ids)}")
            shape_binding.update(
                indices=ids,
                labels=[labels[i] for i in ids],
                original_margin_m=[builder.shape_margin[i] for i in ids],
                original_gap_m=[builder.shape_gap[i] for i in ids],
                scope="peg + fixture only; robot/gripper/table collider settings unchanged",
            )
            for i in ids:
                builder.shape_margin[i] = settings.margin_m
                builder.shape_gap[i] = settings.gap_m
            atomic_write_json(output_dir / "shape_binding.json", shape_binding)

        def configure_solver_attributes(*_args, **_kwargs):
            # MuJoCo Warp can resynchronize options from Newton's per-world
            # custom attributes. Bind those as well as constructor configuration,
            # before solver/graph construction; never suppress a failed readback.
            attributes = NewtonManager._model.mujoco
            original = {
                "iterations": attributes.iterations.numpy().tolist(),
                "tolerance": attributes.tolerance.numpy().tolist(),
            }
            attributes.iterations.fill_(settings.iterations)
            attributes.tolerance.fill_(settings.tolerance)
            atomic_write_json(
                output_dir / "solver_binding.json",
                {
                    "original": original,
                    "applied": {
                        "iterations": attributes.iterations.numpy().tolist(),
                        "tolerance": attributes.tolerance.numpy().tolist(),
                    },
                },
            )

        env = PrecisionEnv(cfg)
        model = NewtonManager._model
        solver_options = NewtonManager._solver.mjw_model.opt
        actual_iterations = int(solver_options.iterations)
        actual_tolerances = solver_options.tolerance.numpy()
        if actual_iterations != settings.iterations or not np.allclose(
            actual_tolerances, settings.tolerance, rtol=1e-5, atol=1e-12
        ):
            raise RuntimeError(
                f"MuJoCo Warp solver option readback differs: requested iterations={settings.iterations}, "
                f"tolerance={settings.tolerance}; got iterations={actual_iterations}, tolerance={actual_tolerances.tolist()}"
            )
        shape_margins = model.shape_margin.numpy()[shape_binding["indices"]]
        shape_gaps = model.shape_gap.numpy()[shape_binding["indices"]]
        if not np.allclose(shape_margins, settings.margin_m, rtol=1e-5, atol=1e-10) or not np.allclose(
            shape_gaps, settings.gap_m, rtol=1e-5, atol=1e-10
        ):
            raise RuntimeError("Newton shape margin/gap readback differs; no silent setting fallback")
        if abs(env.physics_dt - settings.dt_s) > 1e-12 or abs(env.step_dt - 1 / request["policy_hz"]) > 1e-12:
            raise RuntimeError("Runtime timestep/command rate readback mismatch")
        if (
            cfg.sim.physics.solver_cfg.iterations != settings.iterations
            or cfg.sim.physics.solver_cfg.tolerance != settings.tolerance
        ):
            raise RuntimeError("Solver configuration differs from request")

        def reset():
            env.capture_terminal = False
            obs, _ = env.reset(seed=952705)
            env.capture_terminal = True
            env.qualification_wall = torch.zeros(env.num_envs, device=env.device)
            env.qualification_slip = torch.zeros(env.num_envs, device=env.device)
            return obs

        trace, artifacts = [], ["experiment_spec.json", "shape_binding.json", "solver_binding.json"]
        diagnostic_phases = {}
        if request["operation"] == "probes":
            probe_depths = {}
            probe_wall = torch.zeros(env.num_envs, device=env.device)
            for name, offset in (("bore", 0.0), ("rim", 0.020)):
                reset()
                q = tensor(env.robot.data.default_joint_pos).clone()
                q[:, env.arm_ids[0]] += 0.9
                env.robot.write_joint_position_to_sim_index(position=q)
                env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                env.robot.set_joint_effort_target_index(target=torch.zeros_like(q))
                env.robot.set_joint_position_target_index(target=q)
                pose = torch.zeros(env.num_envs, 7, device=env.device)
                pose[:, :3] = (
                    env.scene.env_origins + env.targets + torch.tensor([offset, 0.0, 0.085], device=env.device)
                )
                pose[:, 3] = 1.0
                env.peg.write_root_pose_to_sim_index(root_pose=pose)
                env.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
                for step in range(round(2 / env.physics_dt)):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    values = env.metrics()
                    if name == "bore":
                        probe_wall = torch.maximum(probe_wall, values["wall_violation"])
                    if step % cfg.decimation == 0:
                        trace.append(
                            {
                                "probe": name,
                                "t_s": step * env.physics_dt,
                                "depth_m": values["depth"].cpu().tolist(),
                                "wall_bound_m": values["wall_violation"].cpu().tolist(),
                            }
                        )
                probe_depths[name] = values["depth"].cpu().tolist()
            rows = [
                {
                    "id": key,
                    "depth_m": probe_depths["bore"][i],
                    "rim_depth_m": probe_depths["rim"][i],
                    "valid": np.isfinite(probe_depths["bore"][i]).item(),
                    "hold_s": 0.0,
                    "max_wall_penetration_m": float(probe_wall[i]),
                    "max_grasp_displacement_m": 0.0,
                }
                for i, key in enumerate(request["trials"])
            ]
        else:
            obs = reset()
            actor = None
            if request["operation"] == "frozen_policy":
                from rsl_rl.models import MLPModel
                from tensordict import TensorDict

                checkpoint = torch.load(binding["checkpoint"], map_location="cpu", weights_only=False)
                source, prepared = Path(binding["source"]), Path(binding["prepared"])
                paths = sorted(source.glob("*.py")) + [source / "experiment.json"]
                paths += sorted(prepared.glob("*.usd")) + [prepared / "prepared.json"]
                fingerprint = hashlib.sha256("".join(p.name + sha256_file(p) for p in paths).encode()).hexdigest()
                if checkpoint["infos"]["experiment_fingerprint"] != fingerprint:
                    raise ValueError("Checkpoint source/asset lineage mismatch")
                ppo = read(binding["ppo_config"])
                actor_cfg = copy.deepcopy(ppo["actor"])
                actor_cfg.pop("class_name", None)
                actor_cfg["distribution_cfg"]["class_name"] = "GaussianDistribution"
                observation = TensorDict(obs, batch_size=[env.num_envs], device=env.device)
                actor = MLPModel(observation, ppo["obs_groups"], "actor", 6, **actor_cfg).to(env.device)
                actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                actor.eval()
            if request["video"]:
                import imageio.v2 as imageio
                from PIL import Image, ImageDraw
                from pxr import Gf, UsdGeom

                camera = env.observer_cameras[0]
                x, y, _ = original_job["target_world_m"]
                origin = Gf.Vec3d(*env.scene.env_origins[0].cpu().tolist())
                eye, look = Gf.Vec3d(x + 0.46, y - 0.60, 1.18), Gf.Vec3d(x, y, 0.845)
                xform = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath("/World/ComparisonCamera_0"))
                xform.ClearXformOpOrder()
                xform.AddTransformOp().Set(
                    Gf.Matrix4d().SetLookAt(origin + eye, origin + look, Gf.Vec3d(0, 0, 1)).GetInverse()
                )
                camera.reset()
                for _ in range(12):
                    env.sim.render()
                    camera.update(env.step_dt, force_recompute=True)
                video_path = output_dir / "rollout.mp4"
                writer = imageio.get_writer(str(video_path), fps=15, codec="libx264", macro_block_size=1)
                writers.append(writer)
                artifacts.append(video_path.name)
            target_ee = env.ee_pose().clone()
            peg_pose = tensor(env.peg.data.root_pose_w).clone()
            desired = torch.zeros(env.num_envs, 4, device=env.device)
            desired[:, 0] = 1.0
            orientation = quat_mul(quat_mul(desired, quat_conjugate(peg_pose[:, 3:])), target_ee[:, 3:])
            commanded_depth = torch.ones(env.num_envs, device=env.device) * -0.005
            hold = torch.zeros(env.num_envs, device=env.device)
            outcomes = {}
            phases_passed = {"hold": True, "free_motion": True}
            diagnostic_phases = (
                {
                    phase: {"completed_trial_ids": [], "max_wall_penetration_m": 0.0, "max_grasp_displacement_m": 0.0}
                    for phase in phases_passed
                }
                if actor is None
                else {}
            )
            for step in range(round(30 * request["policy_hz"])):
                time_s = step * env.step_dt
                phase = (
                    "policy"
                    if actor is not None
                    else "hold"
                    if time_s < 1
                    else "free_motion"
                    if time_s < 3
                    else "insert"
                )
                with torch.inference_mode():
                    if actor is not None:
                        observation = TensorDict(obs, batch_size=[env.num_envs], device=env.device)
                        action = actor(observation, stochastic_output=False).clamp(-1, 1)
                    else:
                        ee = env.ee_pose()
                        pose = tensor(env.peg.data.root_pose_w)
                        axis = quat_apply(
                            pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
                        )
                        tip = pose[:, :3] + 0.075 * axis - env.scene.env_origins
                        action = torch.zeros(env.num_envs, 6, device=env.device)
                        target = target_ee.clone()
                        if phase == "free_motion":
                            # Smooth 5 mm lateral out-and-back, safely above the rim in the pinned starts.
                            target[:, 0] += 0.005 * (1 - np.cos(np.pi * (time_s - 1))) / 2
                        pos_error, rot_error = compute_pose_error(
                            ee[:, :3], ee[:, 3:], target[:, :3], target[:, 3:], rot_error_type="axis_angle"
                        )
                        action[:, :3] = pos_error / torch.tensor([0.001, 0.001, 0.002], device=env.device)
                        action[:, 3:] = rot_error / 0.015
                        if phase == "insert":
                            _, rotation_error = compute_pose_error(
                                ee[:, :3], ee[:, 3:], ee[:, :3], orientation, rot_error_type="axis_angle"
                            )
                            action[:, 3:] = rotation_error / 0.015
                            midline = tip[:, :2] - axis[:, :2] * gates["insertion_depth_m"] / 2
                            action[:, :2] = (env.targets[:, :2] - midline) / 0.001
                            far = tip[:, :2] - axis[:, :2] * gates["insertion_depth_m"]
                            error = torch.maximum(
                                (tip[:, :2] - env.targets[:, :2]).norm(dim=-1), (far - env.targets[:, :2]).norm(dim=-1)
                            )
                            ready = error < geometry["minimum_radial_clearance_m"] * 0.5
                            tracking = (commanded_depth - (0.79 - tip[:, 2])).abs() < 0.0005
                            commanded_depth = torch.where(
                                ready & tracking,
                                (commanded_depth + 0.002 * env.step_dt).clamp_max(0.037),
                                commanded_depth,
                            )
                            action[:, 2] = (0.79 - commanded_depth - tip[:, 2]) / 0.002
                        action = action.clamp(-1, 1)
                    if not bool(torch.isfinite(action).all()):
                        raise ValueError("Non-finite controller/policy action")
                    if outcomes:
                        action[list(outcomes)] = 0
                    obs, _, _, _ = env.step(action)
                    metrics = env.metrics()
                    hold = torch.where(metrics["seated"], hold + env.step_dt, 0)
                    if phase in phases_passed and bool(metrics["invalid"].any()):
                        phases_passed[phase] = False
                    if phase in diagnostic_phases:
                        values = diagnostic_phases[phase]
                        values["max_wall_penetration_m"] = max(
                            values["max_wall_penetration_m"], float(env.qualification_wall.max())
                        )
                        values["max_grasp_displacement_m"] = max(
                            values["max_grasp_displacement_m"], float(env.qualification_slip.max())
                        )
                        end = 1.0 if phase == "hold" else 3.0
                        if (step + 1) * env.step_dt >= end - 1e-9:
                            values["completed_trial_ids"] = [
                                key
                                for i, key in enumerate(request["trials"])
                                if i not in outcomes and not bool(metrics["invalid"][i])
                            ]
                    for i, key in enumerate(request["trials"]):
                        if i not in outcomes and (
                            bool(metrics["invalid"][i])
                            or float(hold[i]) >= gates["hold_s"]
                            or step == round(30 * request["policy_hz"]) - 1
                        ):
                            outcomes[i] = {
                                "id": key,
                                "depth_m": float(metrics["depth"][i]),
                                "hold_s": float(hold[i]),
                                "valid": not bool(metrics["invalid"][i]),
                                "max_wall_penetration_m": float(env.qualification_wall[i]),
                                "max_grasp_displacement_m": float(env.qualification_slip[i]),
                                "policy_success": actor is not None and float(hold[i]) >= gates["hold_s"],
                                "terminal_phase": phase,
                                "time_s": (step + 1) * env.step_dt,
                            }
                    trace.append(
                        {
                            "t_s": (step + 1) * env.step_dt,
                            "phase": phase,
                            "action": action.cpu().tolist(),
                            "peg_pose": tensor(env.peg.data.root_pose_w).cpu().tolist(),
                            "depth_m": metrics["depth"].cpu().tolist(),
                            "wall_bound_m": metrics["wall_violation"].cpu().tolist(),
                            "grasp_displacement_m": metrics["slip"].cpu().tolist(),
                        }
                    )
                    if request["video"] and step % 2 == 0:
                        env.sim.render()
                        camera.update(env.step_dt, force_recompute=True)
                        frame = Image.fromarray(tensor(camera.data.output["rgb"])[0].cpu().numpy()[..., :3])
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 60), fill="black")
                        draw.text(
                            (14, 12), f"{binding['hole']} | {phase} | no retraining | simulation only", fill="white"
                        )
                        draw.text(
                            (14, 34),
                            f"t={time_s:.2f}s | first preselected trial {request['trials'][0]} | depth {float(metrics['depth'][0]) * 1000:.2f} mm",
                            fill="white",
                        )
                        writer.append_data(np.asarray(frame))
                    if len(outcomes) == env.num_envs:
                        break
            rows = [outcomes[i] for i in range(env.num_envs)]
            if actor is None:
                for row in rows:
                    row["valid"] = row["valid"] and all(phases_passed.values())
        atomic_write_json(output_dir / "trace.json", trace)
        artifacts.append("trace.json")
        result = {
            "trials": rows,
            "diagnostic_phases": diagnostic_phases,
            "artifacts": artifacts,
            "readback": {
                "shape_margin_m": [float(shape_margins.min()), float(shape_margins.max())],
                "shape_gap_m": [float(shape_gaps.min()), float(shape_gaps.max())],
                "dt_s": env.physics_dt,
                "command_rate_hz": 1 / env.step_dt,
                "solver_configuration": {"iterations": actual_iterations, "tolerance": actual_tolerances.tolist()},
            },
            "measurement_scope": "conservative cylinder/bore envelope at physics cadence; contact force accuracy not assessed",
        }
        # Kit's fast shutdown can terminate the process instead of returning
        # from app.close(). Publish completed, closed artifacts BEFORE shutdown.
        for writer in writers:
            writer.close()
        writers.clear()
        publish(result)
        return result
    except BaseException as error:
        atomic_write_json(output_dir / "physics_failure.json", {"type": type(error).__name__, "message": str(error)})
        traceback.print_exc()
        raise
    finally:
        for writer in writers:
            writer.close()
        if env is not None:
            env.close()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binding, request = read(args.binding), read(args.request)
    if args.output.exists():
        raise FileExistsError("Worker output exists; keep prior attempts")
    if request["operation"] == "describe":
        atomic_write_json(args.output, describe(args.binding, binding))
        return
    if request["operation"] not in {"geometry", "probes", "controlled_insertion", "frozen_policy"}:
        raise ValueError("Unsupported operation; training and hardware are not exposed")
    hashes = {k: sha256_file(p) for k, p in input_files(args.binding, binding).items()}
    if hashes != request["input_hashes"]:
        raise ValueError("Worker input bytes differ from locked plan")

    def publish(result):
        if {k: sha256_file(p) for k, p in input_files(args.binding, binding).items()} != hashes:
            raise ValueError("Input changed during experiment")
        atomic_write_json(
            args.output,
            {
                **result,
                "source": "simulation",
                "input_hashes": hashes,
                "policy_hz": request["policy_hz"],
                "applied_settings": request["settings"],
            },
        )

    if request["operation"] == "geometry":
        geometry = inspect_geometry(binding)
        atomic_write_json(args.output.parent / "geometry.json", geometry)
        publish({**geometry, "artifacts": ["geometry.json"]})
    else:
        physics(binding, request, args.output.parent, publish)


if __name__ == "__main__":
    main()
