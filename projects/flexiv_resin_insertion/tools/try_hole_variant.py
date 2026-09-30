"""Independent actual-Newton probes and frozen-policy trials on the 25.75 hole."""

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--prepared", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--variant", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.backend = "newton"
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "result.json").exists():
    raise FileExistsError("Keep earlier results; use a new output directory")


def dump(name, data):
    (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


dump("status.json", {"state": "starting"})
launcher = AppLauncher(args)
app = launcher.app
env = None
try:
    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    source = Path("/work/flexiv_resin_insertion")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(Path(__file__).parent))
    from hole_variant import environment_class, spec_hash
    from task import ResinPegCfg, tensor

    spec = json.loads(args.variant.read_text())
    assert spec["prepared_sha256"] == hashlib.sha256((args.prepared / "prepared.json").read_bytes()).hexdigest()
    assert spec["hole_usd_sha256"] == hashlib.sha256((args.prepared / "HoleBlock.usd").read_bytes()).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    paths = sorted(source.glob("*.py")) + [source / "experiment.json"]
    paths += sorted(args.prepared.glob("*.usd")) + [args.prepared / "prepared.json"]
    fingerprint = hashlib.sha256(
        "".join(p.name + hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()
    ).hexdigest()
    assert checkpoint["infos"]["experiment_fingerprint"] == fingerprint
    assert checkpoint["infos"]["curriculum_stage"] == spec["curriculum_stage"]
    cfg = ResinPegCfg()
    cfg.seed = 932701
    cfg.prepared_dir = str(args.prepared)
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.evaluation = True
    cfg.curriculum_stage = spec["curriculum_stage"]
    env = environment_class(args.variant)(cfg)
    dump("status.json", {"state": "fixture_probes"})
    probes = []
    for name, offset, required in [
        ("centered_bore", 0.0, True),
        ("chamfer_lead_in", 0.001, False),
        ("solid_rim", 0.020, True),
    ]:
        env.reset(seed=932701)
        q = tensor(env.robot.data.default_joint_pos).clone()
        q[:, env.arm_ids[0]] += 0.9
        env.robot.write_joint_position_to_sim_index(position=q)
        env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        env.robot.set_joint_effort_target_index(target=torch.zeros_like(q))
        env.robot.set_joint_position_target_index(target=q)
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, :3] = env.scene.env_origins + env.target + torch.tensor([offset, 0.0, 0.085], device=env.device)
        pose[:, 3] = 1.0
        env.peg.write_root_pose_to_sim_index(root_pose=pose)
        env.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
        for _ in range(round(2 / env.physics_dt)):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        depth = env.metrics()["depth"]
        passed = bool((depth < 0.003).all()) if name == "solid_rim" else bool(((depth > 0.037) & (depth < 0.041)).all())
        probes.append(
            {"name": name, "offset_m": offset, "depth_m": depth.cpu().tolist(), "passed": passed, "required": required}
        )
        print(json.dumps(probes[-1]), flush=True)
    dump("fixture_probes.json", probes)
    if not all(r["passed"] for r in probes if r["required"]):
        raise RuntimeError("New target bore/rim probe failed; policy trials blocked")

    observations, _ = env.reset(seed=932701)
    env.capture_terminal = True
    obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
    ppo = json.loads((args.run / "ppo_config.json").read_text())
    actor_cfg = copy.deepcopy(ppo["actor"])
    actor_cfg.pop("class_name", None)
    actor_cfg["distribution_cfg"]["class_name"] = "GaussianDistribution"
    actor = MLPModel(obs, ppo["obs_groups"], "actor", 6, **actor_cfg).to(env.device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    outcomes, rows = {}, []
    maximum = torch.full((env.num_envs,), -1.0, device=env.device)
    initial = tensor(env.peg.data.root_pose_w).cpu().tolist()
    updates = int(checkpoint["iter"]) + 1
    dump("status.json", {"state": "policy_trials", "updates": updates})
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            action = actor(obs, stochastic_output=False).clamp(-1, 1)
            if outcomes:
                action[sorted(outcomes)] = 0
            observations, _, terminated, truncated, _ = env.step(action)
            obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
            maximum = torch.maximum(maximum, env.m["depth"])
            for i in range(env.num_envs):
                if i not in outcomes and bool(terminated[i] or truncated[i]):
                    outcomes[i] = {
                        "env_id": i,
                        "success": bool(env.success[i]),
                        "invalid": bool(env.invalid[i]),
                        "timeout": bool(truncated[i]),
                        "time_s": (step + 1) * env.step_dt,
                        "max_depth_mm": float(maximum[i]) * 1000,
                        **{k: float(env.m[k][i]) for k in ["depth", "radial", "tilt", "slip", "wall_violation"]},
                    }
                    print(json.dumps(outcomes[i]), flush=True)
            if step % 3 == 0:
                rows.append(
                    {
                        "time_s": (step + 1) * env.step_dt,
                        "action": action.cpu().tolist(),
                        "peg_pose": tensor(env.peg.data.root_pose_w).cpu().tolist(),
                        "depth_m": env.m["depth"].cpu().tolist(),
                    }
                )
            if len(outcomes) == env.num_envs:
                break
            if time.monotonic() - started > 900:
                raise TimeoutError("Policy trials exceeded 15 minutes")
    result = {
        "target_label": "25.75 with 2 mm chamfer",
        "target_world_m": spec["target_world_m"],
        "variant_sha256": spec_hash(args.variant),
        "base_experiment_fingerprint": fingerprint,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_updates": updates,
        "curriculum_stage": spec["curriculum_stage"],
        "seed": 932701,
        "controller": "frozen learned PPO actor; no scripted commands; no retraining",
        "changed": "task target, held-out IK starts, and geometry-aware validation; fixture/physics/weights unchanged",
        "source_training_modified": False,
        "fixture_probes": probes,
        "initial_peg_poses": initial,
        "outcomes": list(outcomes.values()),
        "successes": sum(x["success"] for x in outcomes.values()),
        "trials": len(outcomes),
        "scope": "four simulated exploratory trials, not a validated success rate or real transfer",
    }
    dump("result.json", result)
    dump("trajectories.json", rows)
    dump("status.json", {"state": "complete", "successes": result["successes"], "trials": result["trials"]})
except BaseException as error:
    (args.output / "failure.txt").write_text(traceback.format_exc())
    dump("status.json", {"state": "failed", "error": str(error)})
    raise
finally:
    if env is not None:
        env.close()
    app.close()
