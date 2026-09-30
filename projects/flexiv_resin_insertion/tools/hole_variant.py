"""Isolated target-hole experiment; never changes the running training task."""

import hashlib
import json
from pathlib import Path

import numpy as np


def spec_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chamfer_penetration(points, spec, xp=np):
    """Sampled screen for the selected bore and its conical entry, in meters.

    The actual Newton collider remains the unmodified triangle mesh. This screen
    is a conservative analytic supplement, not a force or exact mesh certificate.
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    top, bottom = 0.79, 0.75
    distance = None
    for hole in spec.get("all_holes", [spec]):
        dx, dy = x - hole["target_world_m"][0], y - hole["target_world_m"][1]
        radial = xp.sqrt(dx * dx + dy * dy)
        chamfer_start = top - hole["chamfer_height_m"]
        radius = hole["bore_radius_m"] + xp.minimum(
            xp.maximum(z - chamfer_start, xp.zeros_like(z)), xp.ones_like(z) * hole["chamfer_height_m"]
        )
        scale = xp.where(z > chamfer_start, xp.ones_like(z) * np.sqrt(2), xp.ones_like(z))
        candidate = (radial - radius) / scale
        distance = candidate if distance is None else xp.minimum(distance, candidate)
    distance = xp.minimum(distance, xp.minimum(top - z, z - bottom))
    distance = xp.minimum(distance, xp.minimum(0.0635 - xp.abs(x - 0.15), 0.0635 - xp.abs(y)))
    return xp.maximum(distance, xp.zeros_like(distance))


def environment_class(spec_path):
    """Keep the fixture fixed; move initial robot/peg poses and the task target."""
    import math

    import torch
    from isaaclab.utils.math import matrix_from_quat, quat_apply
    from task import ResinPegEnv, tensor

    spec = json.loads(Path(spec_path).read_text())

    class Hole2575Env(ResinPegEnv):
        def __init__(self, cfg, **kwargs):
            if cfg.curriculum_stage != spec["curriculum_stage"]:
                raise ValueError("Variant IK bank does not cover the checkpoint's curriculum stage")
            super().__init__(cfg, **kwargs)
            self.target = torch.tensor(spec["target_world_m"], device=self.device)
            rows = spec["heldout_starts"]
            self.banks[self.curriculum_stage] = {
                key: torch.tensor([r[key] for r in rows], device=self.device, dtype=torch.float32)
                for key in ("joint_pos", "peg_pos", "peg_quat_xyzw", "gripper_pos", "gripper_quat_xyzw")
            }
            self.capture_terminal = False

        def _reset_idx(self, env_ids):
            if getattr(self, "capture_terminal", False):
                return
            super()._reset_idx(env_ids)

        def metrics(self):
            pose = tensor(self.peg.data.root_pose_w)
            local = pose[:, :3] - self.scene.env_origins
            axis = quat_apply(pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
            tip = local + 0.075 * axis
            radial = (tip[:, :2] - self.target[:2]).norm(dim=-1)
            tilt = torch.acos((-axis[:, 2]).clamp(-1, 1))
            depth = self.target[2] - tip[:, 2]
            samples = self.peg_samples.unsqueeze(0).expand(self.num_envs, -1, -1)
            quats = pose[:, None, 3:].expand(-1, samples.shape[1], -1)
            surface = local[:, None, :] + quat_apply(quats.reshape(-1, 4), samples.reshape(-1, 3)).reshape_as(samples)
            wall = chamfer_penetration(surface, spec, xp=torch).amax(dim=1)
            ee = self.ee_pose()
            expected = ee[:, :3] + quat_apply(ee[:, 3:], self.grasp_offset.expand(self.num_envs, -1))
            slip = (pose[:, :3] - expected).norm(dim=-1)
            invalid = (wall > 0.0002) | (surface[:, :, 2].amin(dim=1) < 0.7498) | (slip > 0.015)
            invalid |= ~torch.isfinite(pose).all(dim=-1)
            # Same completion thresholds as the original task; only target and
            # measured free-space geometry change. No relaxed success gate.
            seated = (depth >= 0.035) & (radial <= 0.00035) & (tilt <= math.radians(1)) & ~invalid
            return {
                "radial": radial,
                "tilt": tilt,
                "depth": depth,
                "slip": slip,
                "wall_violation": wall,
                "invalid": invalid,
                "seated": seated,
            }

        def _get_observations(self):
            ee = self.ee_pose()
            hole = self.scene.env_origins + self.target
            rotation = matrix_from_quat(ee[:, 3:])[:, :, :2].reshape(self.num_envs, 6)
            return {
                "policy": torch.cat(
                    [
                        tensor(self.robot.data.joint_pos)[:, self.arm_ids],
                        tensor(self.robot.data.joint_vel)[:, self.arm_ids],
                        ee[:, :3] - hole,
                        rotation,
                        self.actions,
                        tensor(self.robot.data.joint_pos)[:, self.grip_ids[:3]],
                    ],
                    dim=-1,
                )
            }

    return Hole2575Env
