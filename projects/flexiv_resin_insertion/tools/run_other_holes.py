"""Sixteen frozen-policy episodes and four preselected actual-Newton videos."""

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ["prepared", "run", "checkpoint", "spec", "output"]:
    parser.add_argument(f"--{name}", type=Path, required=True)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.backend, args.enable_cameras = "newton", True
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "status.json").exists():
    raise FileExistsError("Preserve every attempt; use a new output directory")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, data):
    (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


dump("status.json", {"state": "starting"})
launcher = AppLauncher(args)
app = launcher.app
env = None
writers = {}
try:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, UsdGeom
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    source = Path("/work/flexiv_resin_insertion")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(Path(__file__).parent))
    from other_hole_env import assignments, environment_class
    from task import ResinPegCfg, tensor

    spec = json.loads(args.spec.read_text())
    assert sha(args.prepared / "prepared.json") == spec["prepared_sha256"]
    assert sha(args.prepared / "HoleBlock.usd") == spec["hole_usd_sha256"]
    assert sha(args.checkpoint) == spec["checkpoint_sha256"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    paths = sorted(source.glob("*.py")) + [source / "experiment.json"]
    paths += sorted(args.prepared.glob("*.usd")) + [args.prepared / "prepared.json"]
    fingerprint = hashlib.sha256("".join(p.name + sha(p) for p in paths).encode()).hexdigest()
    assert checkpoint["infos"]["experiment_fingerprint"] == fingerprint
    assert checkpoint["infos"]["curriculum_stage"] == spec["curriculum_stage"]
    assert all(torch.isfinite(t).all() for t in checkpoint["actor_state_dict"].values())
    layout = assignments(spec)
    selected = [i for i, a in enumerate(layout) if a["record"]]
    cfg = ResinPegCfg()
    cfg.seed = 932701
    cfg.prepared_dir = str(args.prepared)
    cfg.scene.num_envs = len(layout)
    cfg.sim.device = args.device
    cfg.evaluation, cfg.capture = True, False  # Four global observers are created by the subclass.
    cfg.curriculum_stage = spec["curriculum_stage"]
    env = environment_class(args.spec, capture=True)(cfg)

    # Diagnostics are reported independently. We still show exploratory policy
    # behavior if a tight bore cannot pass the unchanged runtime's drop test.
    dump("status.json", {"state": "fixture_probes"})
    probes = {}
    for name, offset in [("centered_bore", 0.0), ("solid_rim", 0.020)]:
        env.reset(seed=932701)
        q = tensor(env.robot.data.default_joint_pos).clone()
        q[:, env.arm_ids[0]] += 0.9
        env.robot.write_joint_position_to_sim_index(position=q)
        env.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        env.robot.set_joint_effort_target_index(target=torch.zeros_like(q))
        env.robot.set_joint_position_target_index(target=q)
        pose = torch.zeros(env.num_envs, 7, device=env.device)
        pose[:, :3] = env.scene.env_origins + env.targets + torch.tensor([offset, 0.0, 0.085], device=env.device)
        pose[:, 3] = 1.0  # XYZW, 180 degrees about X: peg tip down.
        env.peg.write_root_pose_to_sim_index(root_pose=pose)
        env.peg.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
        for _ in range(round(2 / env.physics_dt)):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        depth = env.metrics()["depth"].cpu().tolist()
        probes[name] = [
            {"depth_m": d, "passed": d < 0.003 if name == "solid_rim" else 0.037 < d < 0.041} for d in depth
        ]
        dump("fixture_probes.json", probes)
        print(json.dumps({"probe": name, "depths_m": depth}), flush=True)

    observations, _ = env.reset(seed=932701)
    env.capture_terminal = True
    initial = tensor(env.peg.data.root_pose_w).cpu().numpy()
    local = initial[:, :3] - env.scene.env_origins.cpu().numpy()
    for i, item in enumerate(layout):
        start = spec["jobs"][item["job_index"]]["starts"][item["start_index"]]
        assert np.allclose(local[i], start["peg_pos"], atol=1e-6, rtol=0)
        assert np.allclose(initial[i, 3:], start["peg_quat_xyzw"], atol=1e-6, rtol=0)

    ppo = json.loads((args.run / "ppo_config.json").read_text())
    actor_cfg = copy.deepcopy(ppo["actor"])
    actor_cfg.pop("class_name", None)
    actor_cfg["distribution_cfg"]["class_name"] = "GaussianDistribution"
    obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
    actor = MLPModel(obs, ppo["obs_groups"], "actor", 6, **actor_cfg).to(env.device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    for j, camera in enumerate(env.observer_cameras):
        target = spec["jobs"][j]["target_world_m"]
        origin = Gf.Vec3d(*env.scene.env_origins[selected[j]].cpu().tolist())
        sx, sy = [(1, -1), (1, 1), (-1, -1), (1, -1)][j]
        eye = Gf.Vec3d(target[0] + sx * 0.46, target[1] + sy * 0.60, 1.18)
        look = Gf.Vec3d(target[0], target[1], 0.845)
        xform = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath(f"/World/ComparisonCamera_{j}"))
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(origin + eye, origin + look, Gf.Vec3d(0, 0, 1)).GetInverse())
        camera.reset()
    for _ in range(12):
        env.sim.render()
        for camera in env.observer_cameras:
            camera.update(env.step_dt, force_recompute=True)

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    fonts = [
        ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
        for size in [25, 18, 21]
    ]
    fps = 15
    for j, job in enumerate(spec["jobs"]):
        writers[j] = imageio.get_writer(
            str(args.output / f"{job['key']}.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1
        )
    composite = imageio.get_writer(
        str(args.output / "other_holes_comparison.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    writers["composite"] = composite
    outcomes, rows, frames = {}, [], {}
    maximum = torch.full((env.num_envs,), -1.0, device=env.device)
    maximum_wall = torch.zeros(env.num_envs, device=env.device)
    video_counts = {j: 0 for j in range(4)}
    composite_count = 0
    video_done = set()

    def draw_frame(j, time_s):
        i = selected[j]
        job = spec["jobs"][j]
        camera = env.observer_cameras[j]
        camera.update(env.step_dt, force_recompute=True)
        pixels = tensor(camera.data.output["rgb"])[0].cpu().numpy()[..., :3].copy()
        image = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((0, 0, 960, 76), fill=(14, 22, 31, 241))
        draw.text((17, 8), job["label"], font=fonts[0], fill="white")
        draw.text(
            (17, 42), "Frozen learned PPO | 732 updates | no retraining | 1x", font=fonts[1], fill=(190, 220, 248)
        )
        draw.rectangle((0, 456, 960, 540), fill=(14, 22, 31, 241))
        draw.text(
            (17, 466),
            f"t={time_s:.2f}s   Depth: {float(env.m['depth'][i]) * 1000:.1f}/35 mm   "
            f"Offset: {float(env.m['radial'][i]) * 1000:.2f} mm",
            font=fonts[2],
            fill="white",
        )
        outcome = outcomes.get(i)
        label = "Running learned policy - simulation only"
        color = (255, 199, 82)
        if outcome:
            label = (
                "INSERTED - task checks passed"
                if outcome["success"]
                else ("STOPPED - validity check" if outcome["invalid"] else "TIMEOUT - not inserted")
            )
            if outcome["success"] and not outcome["screened_success"]:
                label = "Task completed - tight-fit physics not verified"
            label += " | end frame"
            color = (175, 225, 105) if outcome["screened_success"] else (255, 183, 107)
        draw.text((17, 501), label, font=fonts[1], fill=color)
        return np.array(image)

    started = time.monotonic()
    dump("status.json", {"state": "recording_and_evaluating", "episodes": len(layout)})
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            raw_actions = actor(obs, stochastic_output=False)
            if not bool(torch.isfinite(raw_actions).all()):
                raise RuntimeError("Non-finite policy action")
            actions = raw_actions.clamp(-1, 1)
            if outcomes:
                actions[sorted(outcomes)] = 0
            observations, _, terminated, truncated, _ = env.step(actions)
            obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
            maximum = torch.maximum(maximum, env.m["depth"])
            maximum_wall = torch.maximum(maximum_wall, env.m["wall_violation"])
            new_terminal = []
            time_s = (step + 1) * env.step_dt
            for i, item in enumerate(layout):
                if i not in outcomes and bool(terminated[i] or truncated[i]):
                    job = spec["jobs"][item["job_index"]]
                    probe_ok = all(probes[name][i]["passed"] for name in probes)
                    clearance_screen = min(0.0002, job["radial_clearance_m"] / 2)
                    outcomes[i] = {
                        "env_id": i,
                        **item,
                        "hole": job["key"],
                        "success": bool(env.success[i]),
                        "invalid": bool(env.invalid[i]),
                        "timeout": bool(truncated[i]),
                        "time_s": time_s,
                        "max_depth_mm": float(maximum[i]) * 1000,
                        "max_wall_violation_m": float(maximum_wall[i]),
                        "fixture_probes_passed": probe_ok,
                        "additional_wall_screen_m": clearance_screen,
                        "screened_success": bool(env.success[i])
                        and probe_ok
                        and float(maximum_wall[i]) <= clearance_screen,
                        **{key: float(env.m[key][i]) for key in ["depth", "radial", "tilt", "slip", "wall_violation"]},
                    }
                    new_terminal.append(i)
                    print(json.dumps(outcomes[i]), flush=True)
                    dump("outcomes_so_far.json", list(outcomes.values()))
            render_due = step % 2 == 0 or any(i in selected for i in new_terminal)
            if render_due and len(video_done) < len(selected):
                env.sim.render()
                for j, i in enumerate(selected):
                    if j not in video_done:
                        frames[j] = draw_frame(j, time_s)
                        writers[j].append_data(frames[j])
                        video_counts[j] += 1
                        if i in outcomes:
                            for _ in range(2 * fps):
                                writers[j].append_data(frames[j])
                                video_counts[j] += 1
                            writers[j].close()
                            del writers[j]
                            video_done.add(j)
                    if step % 60 == 0 or i in new_terminal:
                        imageio.imwrite(str(args.output / f"{spec['jobs'][j]['key']}_frame_{step:04d}.png"), frames[j])
                combined = np.concatenate(
                    [
                        np.concatenate([frames[0], frames[1]], axis=1),
                        np.concatenate([frames[2], frames[3]], axis=1),
                    ],
                    axis=0,
                )
                composite.append_data(combined)
                composite_count += 1
                if len(video_done) == 4:
                    for _ in range(2 * fps):
                        composite.append_data(combined)
                        composite_count += 1
                    composite.close()
                    del writers["composite"]
            if step % 3 == 0 or new_terminal:
                rows.append(
                    {
                        "time_s": time_s,
                        "applied_actions": actions.cpu().tolist(),
                        "peg_poses": tensor(env.peg.data.root_pose_w).cpu().tolist(),
                        "depth_m": env.m["depth"].cpu().tolist(),
                        "radial_m": env.m["radial"].cpu().tolist(),
                    }
                )
            if step % 30 == 0 or new_terminal:
                dump(
                    "status.json",
                    {
                        "state": "recording_and_evaluating",
                        "sim_time_s": time_s,
                        "completed_episodes": len(outcomes),
                        "videos_complete": len(video_done),
                    },
                )
            if len(outcomes) == len(layout):
                break
            if time.monotonic() - started > 2700:
                raise TimeoutError("Bounded capture/evaluation exceeded 45 minutes")

    assert len(outcomes) == len(layout) and len(video_done) == 4
    summaries = []
    for j, job in enumerate(spec["jobs"]):
        trials = [o for o in outcomes.values() if o["job_index"] == j]
        summaries.append(
            {
                "key": job["key"],
                "label": job["label"],
                "task_passes": sum(o["success"] for o in trials),
                "screened_passes": sum(o["screened_success"] for o in trials),
                "trials": len(trials),
                "recorded_outcome": outcomes[selected[j]],
                "video": f"{job['key']}.mp4",
                "radial_clearance_mm": job["radial_clearance_m"] * 1000,
            }
        )
    result = {
        "status": "complete",
        "controller": "frozen PPO actor; deterministic inference; no scripted actions or retraining",
        "checkpoint_sha256": sha(args.checkpoint),
        "training_updates": 732,
        "base_experiment_fingerprint": fingerprint,
        "spec_sha256": sha(args.spec),
        "tool_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "initial_peg_poses": initial.tolist(),
        "env_origins": env.scene.env_origins.cpu().tolist(),
        "summaries": summaries,
        "outcomes": list(outcomes.values()),
        "fixture_probes": probes,
        "selected_env_ids": selected,
        "selected_source_heldout_index": 14,
        "videos": {
            "fps": fps,
            "nominal_playback_speed": 1,
            "terminal_frame_hold_s": 2,
            "frame_counts": video_counts,
            "composite_frames": composite_count,
            "note": "2x2 panels share simulation time; finished panels explicitly hold their terminal frame",
        },
        "source_training_modified": False,
        "physics_parameters_changed": False,
        "scope": spec["scope"],
        "precision_caveat": spec["precision_caveat"],
        "screening_caveat": "The extra half-clearance sampled-wall screen and free-peg probes are diagnostics, not exact penetration, force, safety, or real-transfer certification.",
    }
    dump("result.json", result)
    dump("trajectories.json", rows)
    dump("status.json", {"state": "complete", "summaries": summaries})
except BaseException as error:
    (args.output / "failure.txt").write_text(traceback.format_exc())
    dump("status.json", {"state": "failed", "error": str(error)})
    raise
finally:
    for writer in writers.values():
        writer.close()
    if env is not None:
        env.close()
    app.close()
