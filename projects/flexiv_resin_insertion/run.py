"""Qualification-gated, checkpointed sim-only insertion experiment entry point."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fingerprint(prepared):
    paths = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).with_name("experiment.json")]
    paths += sorted(Path(prepared).glob("*.usd")) + [Path(prepared) / "prepared.json"]
    return hashlib.sha256(
        "".join(p.name + hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()
    ).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["qualify", "train", "evaluate"], required=True)
parser.add_argument("--prepared", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--capture", action="store_true")
parser.add_argument("--max_wall_hours", type=float, default=48)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.backend = "newton"
if args.capture:
    args.enable_cameras = True
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
experiment_fingerprint = fingerprint(args.prepared)
atomic_json(
    output / "status.json",
    {
        "state": "launching_app",
        "updated_unix": time.time(),
        "experiment_fingerprint": experiment_fingerprint,
        "hardware_execution": False,
    },
)
stop_requested = False


def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
launcher = AppLauncher(args)
app = launcher.app

# AppLauncher adjusts import paths. Restore this standalone experiment's path
# explicitly; imports failing after Kit starts must leave a durable failure.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import torch
    from task import ResinPegCfg, ResinPegEnv, tensor
except BaseException:
    (output / "startup_failure.txt").write_text(traceback.format_exc())
    atomic_json(output / "status.json", {"state": "startup_failed", "hardware_execution": False})
    app.close()
    raise


def status(state, **extra):
    atomic_json(
        output / "status.json",
        {
            "state": state,
            "updated_unix": time.time(),
            "experiment_fingerprint": experiment_fingerprint,
            "hardware_execution": False,
            **extra,
        },
    )
    print("[RESIN] " + state + " " + json.dumps(extra), flush=True)


def agent_config():
    from isaaclab.utils.configclass import configclass
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    from rewards import PPO_GAMMA
    from rl_compat import rsl5_config

    @configclass
    class PPOCfg(RslRlOnPolicyRunnerCfg):
        seed = 20260916
        num_steps_per_env = 128
        max_iterations = args.iterations
        save_interval = 10
        experiment_name = "flexiv_resin_insertion"
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}  # noqa: RUF012 -- Isaac Lab configclass
        actor = RslRlMLPModelCfg(
            hidden_dims=[256, 128, 64],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35),
        )
        critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=0.0003,
            schedule="adaptive",
            gamma=PPO_GAMMA,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    return rsl5_config(PPOCfg().to_dict(), importlib.metadata.version("rsl-rl-lib"))


def fixture_probes(env):
    """Independent gravity drops: bore must admit the peg, solid rim must stop it."""
    env.reset(seed=20260916)
    q = tensor(env.robot.data.default_joint_pos).clone()
    q[:, env.arm_ids[0]] += 0.9
    env.robot.write_joint_position_to_sim_index(position=q)
    env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
    env.robot.set_joint_effort_target_index(target=torch.zeros_like(q))
    env.robot.set_joint_position_target_index(target=q)
    records = []
    for label, offset in [("centered_bore", 0.0), ("solid_rim", 0.017)]:
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, :3] = env.scene.env_origins + torch.tensor([0.15 + offset, 0.0, 0.875], device=env.device)
        pose[:, 3] = 1.0  # XYZW: 180 degrees about X, peg tip down.
        env.peg.write_root_pose_to_sim_index(root_pose=pose)
        env.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
        for _ in range(round(2.0 / env.physics_dt)):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        m = env.metrics()
        depth = m["depth"].cpu().tolist()
        passed = (
            bool(((m["depth"] > 0.037) & (m["depth"] < 0.041)).all())
            if offset == 0
            else bool((m["depth"] < 0.003).all())
        )
        records.append({"probe": label, "depth_m": depth, "passed": passed})
    atomic_json(output / "fixture_probes.json", records)
    print("[RESIN] fixture probes " + json.dumps(records), flush=True)
    env.reset(seed=20260916)
    env.episode_records.clear()
    if not all(r["passed"] for r in records):
        raise RuntimeError("Independent bore/rim physics probe failed; qualification blocked")


def qualify(env):
    from isaaclab.utils.math import compute_pose_error

    fixture_probes(env)
    env.reset(seed=20260916)
    # Numerical Jacobian check against actual runtime FK, before any motion.
    q = tensor(env.robot.data.joint_pos).clone()
    pose0 = env.ee_pose()
    jacobian = tensor(env.robot.data.body_link_jacobian_w)[:, env.ee_id - 1, :, :][:, :, env.arm_ids].clone()
    numeric = torch.zeros_like(jacobian)
    for j, joint_id in enumerate(env.arm_ids):
        shifted = q.clone()
        shifted[:, joint_id] += 0.0005
        env.robot.write_joint_position_to_sim_index(position=shifted)
        shifted_pose = env.ee_pose()
        delta_position, delta_rotation = compute_pose_error(
            pose0[:, :3], pose0[:, 3:], shifted_pose[:, :3], shifted_pose[:, 3:], rot_error_type="axis_angle"
        )
        numeric[:, :, j] = torch.cat([delta_position, delta_rotation], dim=-1) / 0.0005
    env.robot.write_joint_position_to_sim_index(position=q)
    reset_position_error = (pose0[:, :3] - env.reset_ee_pose[:, :3]).norm(dim=-1)
    jacobian_error = float((numeric[:, :3] - jacobian[:, :3]).abs().max())
    angular_error = float((numeric[:, 3:] - jacobian[:, 3:]).abs().max())
    atomic_json(
        output / "controller_geometry_check.json",
        {
            "reset_position_error_m": reset_position_error.cpu().tolist(),
            "max_linear_jacobian_error_m_per_rad": jacobian_error,
            "max_angular_jacobian_error_rad_per_rad": angular_error,
            "analytic": jacobian.cpu().tolist(),
            "finite_difference": numeric.cpu().tolist(),
            "initial_ee_pose": pose0.cpu().tolist(),
            "expected_ee_pose": env.reset_ee_pose.cpu().tolist(),
        },
    )
    if float(reset_position_error.max()) > 0.001 or jacobian_error > 0.01 or angular_error > 0.03:
        raise RuntimeError("Controller geometry/Jacobian validation failed before physics")
    env.reset(seed=20260916)
    rows = []
    self_contact_pairs = set()
    hold_max_slip = 0.0
    hold_invalid = False
    start = time.monotonic()
    video = None
    if args.capture:
        import imageio.v2 as imageio
        from pxr import Gf, UsdGeom

        camera_prim = env.sim.stage.GetPrimAtPath("/World/Camera")
        origin = Gf.Vec3d(*env.scene.env_origins[0].cpu().tolist())
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(origin + Gf.Vec3d(0.65, -0.75, 1.22), origin + Gf.Vec3d(0.15, 0.0, 0.9), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        for _ in range(12):
            env.sim.render()
            env.camera.update(env.step_dt, force_recompute=True)
        video = imageio.get_writer(str(output / "qualification.mp4"), fps=30, codec="libx264", quality=8)
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            inserting = step >= 60
            actions = env.scripted_action(insert=inserting)
            completed_envs = sorted({r["env_id"] for r in env.episode_records if r["success"]})
            if completed_envs:
                actions[completed_envs] = env.scripted_action(insert=False)[completed_envs]
            _, _reward, _terminated, _truncated, _ = env.step(actions)
            m = env.m
            if video is not None:
                env.sim.render()
                env.camera.update(env.step_dt, force_recompute=True)
                pixels = tensor(env.camera.data.output["rgb"])[0].cpu().numpy()[..., :3]
                video.append_data(pixels)
                if step in (0, 59, 120, 210):
                    imageio.imwrite(str(output / f"diagnostic_{step:03d}.png"), pixels)
            if not inserting:
                hold_max_slip = max(hold_max_slip, float(m["slip"].max()))
                hold_invalid |= bool(m["invalid"].any())
            if step % 3 == 0:
                from isaaclab_newton.physics.newton_manager import NewtonManager

                contacts = NewtonManager.get_contacts()
                count = int(contacts.rigid_contact_count.numpy()[0])
                labels = NewtonManager._model.shape_label
                for ia, ib in zip(
                    contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
                ):
                    if ia >= 0 and ib >= 0 and "/Robot/" in labels[ia] and "/Robot/" in labels[ib]:
                        self_contact_pairs.add(tuple(sorted((labels[ia], labels[ib]))))
                rows.append(
                    {
                        "step": step,
                        "time_s": (step + 1) * env.step_dt,
                        "phase": "insert" if inserting else "hold",
                        "depth_m": m["depth"].cpu().tolist(),
                        "slip_m": m["slip"].cpu().tolist(),
                        "radial_m": m["radial"].cpu().tolist(),
                        "tilt_rad": m["tilt"].cpu().tolist(),
                        "wall_violation_m": m["wall_violation"].cpu().tolist(),
                        "joint_pos_rad": tensor(env.robot.data.joint_pos).cpu().tolist(),
                        "ee_pose": env.ee_pose().cpu().tolist(),
                        "peg_pose": tensor(env.peg.data.root_pose_w).cpu().tolist(),
                        "ee_target": env.goal_pose.cpu().tolist(),
                        "velocity_consistency_error": env.velocity_consistency_error.cpu().tolist(),
                        "commanded_torque_nm": env.commanded_tau.cpu().tolist(),
                    }
                )
            if stop_requested:
                raise InterruptedError("Stop requested during qualification")
            if any(r["invalid"] for r in env.episode_records):
                break
            if len({r["env_id"] for r in env.episode_records if r["success"]}) == env.num_envs:
                break
    if video is not None:
        video.close()
    successes = {r["env_id"] for r in env.episode_records if r["success"]}
    invalid_episodes = sum(r["invalid"] for r in env.episode_records)
    passed = not hold_invalid and hold_max_slip < 0.005 and len(successes) == env.num_envs and invalid_episodes == 0
    record = {
        "passed": passed,
        "experiment_fingerprint": experiment_fingerprint,
        "num_envs": env.num_envs,
        "hold_max_slip_m": hold_max_slip,
        "hold_invalid": hold_invalid,
        "successful_envs": sorted(successes),
        "invalid_episodes": invalid_episodes,
        "episodes": env.episode_records,
        "robot_self_contact_candidate_pairs": sorted(self_contact_pairs),
        "max_velocity_consistency_error": max(max(r["velocity_consistency_error"]) for r in rows),
        "elapsed_s": time.monotonic() - start,
        "controller": "scripted Cartesian diagnostic, NOT trained policy",
        "scope": "simulated grasp retention and nominal insertion; no real transfer claim",
    }
    atomic_json(output / "qualification_trajectories.json", rows)
    atomic_json(output / "qualification.json", record)
    status("qualified" if passed else "qualification_failed", **record)
    if not passed:
        raise RuntimeError("Physical qualification failed; long training is blocked. See qualification.json")


def training(env):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    cfg = agent_config()

    class DurableRunner(OnPolicyRunner):
        def save(self, path, infos=None):
            temporary = str(path) + ".partial"
            super().save(
                temporary,
                infos={
                    "experiment_fingerprint": experiment_fingerprint,
                    "curriculum_stage": env.curriculum_stage,
                    "resume_semantics": "policy+optimizer; episodes restart",
                },
            )
            os.replace(temporary, path)
            atomic_json(
                output / "latest_checkpoint.json",
                {
                    "path": str(path),
                    "completed_iteration": self.current_learning_iteration,
                    "experiment_fingerprint": experiment_fingerprint,
                },
            )

    runner = DurableRunner(wrapped, cfg, log_dir=str(output / "checkpoints"), device=env.device)
    completed = 0
    checkpoint = output / "latest_checkpoint.json"
    if args.resume and checkpoint.exists():
        metadata = json.loads(checkpoint.read_text())
        if metadata["experiment_fingerprint"] != experiment_fingerprint:
            raise ValueError("Refusing to resume a checkpoint after experiment changes")
        info = runner.load(metadata["path"])
        env.curriculum_stage = info["curriculum_stage"]
        completed = int(metadata["completed_iteration"]) + 1
        runner.current_learning_iteration = completed
    elif checkpoint.exists():
        raise ValueError("Existing checkpoint requires explicit --resume")
    atomic_json(output / "ppo_config.json", cfg)
    start = time.monotonic()
    while completed < args.iterations and not stop_requested:
        chunk = min(10, args.iterations - completed)
        status("training", completed_iterations=completed, curriculum_stage=env.curriculum_stage)
        chunk_start = time.monotonic()
        runner.learn(chunk)
        completed = runner.current_learning_iteration + 1
        runner.current_learning_iteration = completed
        recent = env.episode_records[-200:]
        same_stage = [r for r in recent if r["stage"] == env.curriculum_stage]
        success = sum(r["success"] for r in same_stage) / max(1, len(same_stage))
        invalid = sum(r["invalid"] for r in same_stage) / max(1, len(same_stage))
        if len(same_stage) >= 100 and success >= 0.8 and invalid <= 0.02 and env.curriculum_stage < 2:
            env.curriculum_stage += 1
        record = {
            "completed_iterations": completed,
            "curriculum_stage": env.curriculum_stage,
            "recent_training_success_rate": success,
            "recent_invalid_rate": invalid,
            "chunk_seconds": time.monotonic() - chunk_start,
            "transitions_per_second": chunk
            * cfg["num_steps_per_env"]
            * env.num_envs
            / (time.monotonic() - chunk_start),
        }
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        status("training", **record)
        if time.monotonic() - start >= args.max_wall_hours * 3600:
            status("budget_exhausted", **record)
            return
    status(
        "stopped_checkpointed" if stop_requested else "training_complete_needs_evaluation",
        completed_iterations=completed,
        curriculum_stage=env.curriculum_stage,
    )


def evaluate(env):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    metadata = json.loads((output / "latest_checkpoint.json").read_text())
    if metadata["experiment_fingerprint"] != experiment_fingerprint:
        raise ValueError("Checkpoint and evaluation configuration do not match")
    env.curriculum_stage = 2
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, agent_config(), log_dir=None, device=env.device)
    runner.load(metadata["path"])
    policy = runner.get_inference_policy(device=env.device)
    env.episode_records.clear()
    env.reset(seed=932701)
    observations = wrapped.get_observations()
    start = time.monotonic()
    with torch.inference_mode():
        while len(env.episode_records) < 200:
            observations, _, _, _ = wrapped.step(policy(observations))
            if stop_requested or time.monotonic() - start > 7200:
                raise InterruptedError("Evaluation interrupted or exceeded two-hour limit")
    records = env.episode_records[:200]
    result = {
        "scope": "simulation only; unmeasured resin priors; not real transfer",
        "checkpoint": metadata,
        "heldout_seed": 932701,
        "curriculum_stage": 2,
        "episodes": records,
        "trial_count": len(records),
        "unique_available_initial_poses": len(env.banks[2]["joint_pos"]),
        "repeated_starts": True,
        "independent_real_trials": False,
        "success_rate": sum(r["success"] for r in records) / len(records),
        "invalid_rate": sum(r["invalid"] for r in records) / len(records),
    }
    atomic_json(output / "heldout_evaluation.json", result)
    status("evaluated", success_rate=result["success_rate"], invalid_rate=result["invalid_rate"])


env = None
try:
    if args.mode in {"train", "evaluate"}:
        qualification = json.loads((output / "qualification.json").read_text())
        if not qualification.get("passed") or qualification["experiment_fingerprint"] != experiment_fingerprint:
            raise RuntimeError("Missing/stale physical qualification. Train/evaluate blocked.")
    cfg = ResinPegCfg()
    cfg.seed = 20260916
    cfg.prepared_dir = str(Path(args.prepared).resolve())
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.evaluation = args.mode == "evaluate"
    cfg.capture = args.capture
    status("initializing", mode=args.mode, num_envs=args.num_envs)
    env = ResinPegEnv(cfg)
    versions = {name: importlib.metadata.version(name) for name in ["newton", "torch", "rsl-rl-lib"]}
    from isaaclab_newton.physics.newton_manager import NewtonManager

    atomic_json(
        output / "runtime.json",
        {
            "versions": versions,
            "isaac_lab_release": "3.0.0-beta2",
            "physics": "Newton",
            "solver": "MuJoCo Warp",
            "body_count": NewtonManager._model.body_count,
            "shape_count": NewtonManager._model.shape_count,
            "shape_labels": list(NewtonManager._model.shape_label),
            "shape_types": NewtonManager._model.shape_type.numpy().tolist(),
            "shape_transforms": NewtonManager._model.shape_transform.numpy().tolist(),
            "shape_scales": NewtonManager._model.shape_scale.numpy().tolist(),
            "num_envs": env.num_envs,
            "dt_s": cfg.sim.dt,
            "substeps": cfg.sim.physics.num_substeps,
            "policy_dt_s": env.step_dt,
            "robot_gravity_compensation": "ideal body gravcomp=1; not measured vendor controller",
            "peg_dynamic": True,
            "peg_welded": False,
            "calibrated": False,
        },
    )
    if args.mode == "qualify":
        qualify(env)
    elif args.mode == "train":
        training(env)
    else:
        evaluate(env)
except BaseException as error:
    (output / "failure.txt").write_text(traceback.format_exc())
    status("failed", error=str(error))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
