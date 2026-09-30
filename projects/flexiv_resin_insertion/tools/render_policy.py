"""Read-only learned-policy video using the frozen training image and assets.

This file is mounted outside the fingerprinted training package. It never calls
the scripted controller, modifies a checkpoint, or writes to the training run.
"""

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--run", required=True, type=Path)
parser.add_argument("--prepared", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seed", type=int, default=932701)
parser.add_argument("--variant", type=Path)
parser.add_argument("--trial-record", type=Path)
parser.add_argument("--env-index", type=int, default=0)
parser.add_argument("--playback-speed", type=float, choices=[0.5, 1.0], default=1.0)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if bool(args.variant) != bool(args.trial_record):
    parser.error("Variant capture requires both --variant and --trial-record")
args.backend = "newton"
args.enable_cameras = True
args.output.mkdir(parents=True, exist_ok=True)
if (args.output / "policy_progress.mp4").exists():
    raise FileExistsError("Use a new capture directory; do not overwrite earlier evidence")
(args.output / "capture_status.json").write_text(json.dumps({"state": "starting"}))
launcher = AppLauncher(args)
app = launcher.app
env = None
writer = None


def dump(name, data):
    (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


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
    from task import ResinPegCfg, ResinPegEnv, tensor

    paths = sorted(source.glob("*.py")) + [source / "experiment.json"]
    paths += sorted(args.prepared.glob("*.usd")) + [args.prepared / "prepared.json"]
    fingerprint = hashlib.sha256(
        "".join(p.name + hashlib.sha256(p.read_bytes()).hexdigest() for p in paths).encode()
    ).hexdigest()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    qualification = json.loads((args.run / "qualification.json").read_text())
    assert qualification["passed"] is True
    assert qualification["experiment_fingerprint"] == fingerprint
    assert checkpoint["infos"]["experiment_fingerprint"] == fingerprint
    assert all(torch.isfinite(t).all() for t in checkpoint["actor_state_dict"].values())
    updates = int(checkpoint["iter"]) + 1
    stage = int(checkpoint["infos"]["curriculum_stage"])
    selected = args.env_index
    trial_record = None
    variant = None
    parent_class = ResinPegEnv
    if args.variant:
        from hole_variant import environment_class, spec_hash

        variant = json.loads(args.variant.read_text())
        trial_record = json.loads(args.trial_record.read_text())
        assert trial_record["checkpoint_sha256"] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        assert trial_record["variant_sha256"] == spec_hash(args.variant)
        assert trial_record["base_experiment_fingerprint"] == fingerprint
        assert variant["prepared_sha256"] == hashlib.sha256((args.prepared / "prepared.json").read_bytes()).hexdigest()
        assert variant["hole_usd_sha256"] == hashlib.sha256((args.prepared / "HoleBlock.usd").read_bytes()).hexdigest()
        assert args.seed == trial_record["seed"]
        assert all(p["passed"] for p in trial_record["fixture_probes"] if p["required"])
        assert 0 <= selected < trial_record["trials"]
        parent_class = environment_class(args.variant)
    elif selected != 0:
        raise ValueError("The original single-environment capture only supports env-index 0")

    class TerminalCaptureEnv(parent_class):
        """Keep the terminal physics state visible; never step beyond termination."""

        def _reset_idx(self, env_ids):
            if getattr(self, "capture_terminal", False):
                return  # Only automatic terminal reset; initial reset uses the original implementation.
            super()._reset_idx(env_ids)

    cfg = ResinPegCfg()
    cfg.seed = args.seed
    cfg.prepared_dir = str(args.prepared)
    cfg.scene.num_envs = trial_record["trials"] if trial_record else 1
    cfg.sim.device = args.device
    cfg.evaluation = True
    cfg.curriculum_stage = stage
    cfg.capture = True
    env = TerminalCaptureEnv(cfg)
    if env.num_envs > 1:
        # /World/Camera is one global observer, not one sensor per environment.
        # Scene.reset otherwise passes four environment indices to one camera.
        # Manage only this observer manually; robot/peg resets stay unchanged.
        assert env.scene.sensors.pop("overview") is env.camera
    observations, _ = env.reset(seed=args.seed)
    env.camera.reset()
    env.capture_terminal = True
    initial_poses = tensor(env.peg.data.root_pose_w).cpu().tolist()
    if trial_record and not np.allclose(initial_poses, trial_record["initial_peg_poses"], atol=1e-6, rtol=0):
        raise RuntimeError("Capture starting poses do not reproduce the recorded four-trial initialization")

    # RSL-RL mutates class-name fields when creating models. Reconstitute only
    # the known MLP/Gaussian names from this pinned v24 training architecture.
    ppo_cfg = json.loads((args.run / "ppo_config.json").read_text())
    actor_cfg = copy.deepcopy(ppo_cfg["actor"])
    actor_cfg.pop("class_name", None)
    actor_cfg["distribution_cfg"]["class_name"] = "GaussianDistribution"
    obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
    actor = MLPModel(obs, ppo_cfg["obs_groups"], "actor", 6, **actor_cfg).to(env.device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()

    camera = UsdGeom.Xformable(env.sim.stage.GetPrimAtPath("/World/Camera"))
    origin = Gf.Vec3d(*env.scene.env_origins[selected].cpu().tolist())
    eye = Gf.Vec3d(0.52, 0.64, 1.18) if variant else Gf.Vec3d(0.65, -0.75, 1.22)
    look_at = (
        Gf.Vec3d(variant["target_world_m"][0], variant["target_world_m"][1], 0.845)
        if variant
        else Gf.Vec3d(0.15, 0.0, 0.9)
    )
    camera.ClearXformOpOrder()
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(origin + eye, origin + look_at, Gf.Vec3d(0, 0, 1)).GetInverse())
    for _ in range(12):
        env.sim.render()
        env.camera.update(env.step_dt, force_recompute=True)

    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    font_path = next((p for p in font_candidates if p.exists()), None)

    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)

    title_font, body_font, small_font = font(29), font(23), font(19)

    def frame(row, outcome="Running learned policy"):
        env.sim.render()
        env.camera.update(env.step_dt, force_recompute=True)
        pixels = tensor(env.camera.data.output["rgb"])[0].cpu().numpy()[..., :3].copy()
        canvas = Image.fromarray(pixels).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, 1280, 108), fill=(14, 22, 31, 238))
        title = "Learned policy | 25.75 mm chamfered hole" if variant else "Flexiv peg insertion | Learned PPO policy"
        draw.text((25, 12), title, font=title_font, fill="white")
        draw.text(
            (25, 53),
            f"{updates} training updates  |  Isaac Lab + Newton  |  {args.playback_speed:g}x playback",
            font=body_font,
            fill=(190, 220, 248),
        )
        scope = (
            f"SELECTED EXAMPLE, TRIAL {selected + 1}  |  Prior test: {trial_record['successes']} of {trial_record['trials']} inserted  |  SIMULATION ONLY"
            if trial_record
            else "EARLY CHECKPOINT - NOT SUCCESS-VALIDATED - SIMULATION ONLY"
        )
        draw.text((25, 83), scope, font=small_font, fill=(255, 199, 82))
        draw.rectangle((0, 621, 1280, 720), fill=(14, 22, 31, 238))
        draw.text(
            (25, 633),
            f"Sim time: {row['time_s']:.2f}s    Depth: {row['depth_mm']:.1f} / 35 mm    "
            f"Tip offset: {row['radial_mm']:.2f} mm    Grasp shift: {row['slip_mm']:.1f} mm",
            font=body_font,
            fill="white",
        )
        draw.text((25, 671), outcome, font=body_font, fill=(255, 199, 82))
        return np.array(canvas)

    rows = []
    fps = round(args.playback_speed / env.step_dt)
    writer = imageio.get_writer(
        str(args.output / "policy_progress.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    dump("capture_status.json", {"state": "recording", "updates": updates, "stage": stage})
    started = time.monotonic()
    outcome = "Episode time limit reached - insertion not completed"
    terminal = {}
    finished = set()
    with torch.inference_mode():
        for step in range(env.max_episode_length):
            actions = actor(obs, stochastic_output=False)
            if finished:
                actions[sorted(finished)] = 0
            observations, _reward, terminated, truncated, _ = env.step(actions.clamp(-1, 1))
            obs = TensorDict(observations, batch_size=[env.num_envs], device=env.device)
            m = env.m
            row = {
                "time_s": (step + 1) * env.step_dt,
                "depth_mm": float(m["depth"][selected]) * 1000,
                "radial_mm": float(m["radial"][selected]) * 1000,
                "slip_mm": float(m["slip"][selected]) * 1000,
                "tilt_rad": float(m["tilt"][selected]),
                "wall_violation_mm": float(m["wall_violation"][selected]) * 1000,
                "actions": actions[selected].cpu().tolist(),
                "joint_pos_rad": tensor(env.robot.data.joint_pos)[selected].cpu().tolist(),
                "peg_pose": tensor(env.peg.data.root_pose_w)[selected].cpu().tolist(),
            }
            rows.append(row)
            finished.update(torch.where(terminated | truncated)[0].cpu().tolist())
            done = bool(terminated[selected] or truncated[selected])
            if done:
                terminal = {
                    "success": bool(env.success[selected]),
                    "invalid": bool(env.invalid[selected]),
                    "truncated": bool(truncated[selected]),
                }
                if terminal["success"]:
                    outcome = "INSERTED - depth, alignment and sustained-hold checks passed (simulation)"
                elif terminal["invalid"]:
                    outcome = "Stopped: grasp/contact validity check failed - not a successful insertion"
                pixels = frame(row, outcome)
            else:
                pixels = frame(row)
            writer.append_data(pixels)
            if step % 30 == 0 or done:
                imageio.imwrite(str(args.output / f"preview_{step:04d}.png"), pixels)
                dump(
                    "capture_status.json",
                    {"state": "recording", "frame": step, "sim_time_s": row["time_s"], "depth_mm": row["depth_mm"]},
                )
                print(json.dumps({"frame": step, "time_s": row["time_s"], "depth_mm": row["depth_mm"]}), flush=True)
            if done:
                break
            if time.monotonic() - started > 1800:
                raise TimeoutError("Capture exceeded its 30-minute budget")
    for _ in range(2 * fps):
        writer.append_data(pixels)
    writer.close()
    writer = None
    dump(
        "rollout.json",
        {
            "controller": "learned PPO actor; deterministic inference; no scripted actions",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "experiment_fingerprint": fingerprint,
            "completed_training_updates": updates,
            "curriculum_stage": stage,
            "seed": args.seed,
            "evidence": "selected prior-test starting condition; fresh learned-policy simulation capture"
            if trial_record
            else "one predetermined held-out simulated start; no trial selection",
            "scope": "illustrative simulated rollout, not a success-rate estimate or real transfer",
            "selected_env_index": selected,
            "initial_peg_poses": initial_poses,
            "variant_sha256": spec_hash(args.variant) if variant else None,
            "prior_trial_record_sha256": hashlib.sha256(args.trial_record.read_bytes()).hexdigest()
            if trial_record
            else None,
            "playback_speed": args.playback_speed,
            "fps": fps,
            "render_only_changes": "global RTX observer camera reset/updated manually; suppress terminal automatic reset; 2-second end-frame hold",
            "outcome": outcome,
            "terminal": terminal,
            "frames": rows,
        },
    )
    dump("capture_status.json", {"state": "complete", "outcome": outcome, **terminal})
except BaseException as error:
    (args.output / "capture_failure.txt").write_text(traceback.format_exc())
    dump("capture_status.json", {"state": "failed", "error": str(error)})
    raise
finally:
    if writer is not None:
        writer.close()
    if env is not None:
        env.close()
    app.close()
