"""Matched reset banks and per-environment targets for frozen-policy experiments."""

import json
from pathlib import Path

from hole_variant import chamfer_penetration


def assignments(spec):
    result = []
    for job_index, job in enumerate(spec["jobs"]):
        for start_index, start in enumerate(job["starts"]):
            result.append(
                {
                    "job_index": job_index,
                    "start_index": start_index,
                    "source_heldout_index": start["source_heldout_index"],
                    "record": start_index == job["recorded_start_index"],
                }
            )
    if not result or len({len(j["starts"]) for j in spec["jobs"]}) != 1:
        raise ValueError("Require nonempty, equally sized matched banks")
    if any(sum(r["record"] for r in result if r["job_index"] == j) != 1 for j in range(len(spec["jobs"]))):
        raise ValueError("Exactly one predeclared recorded episode is required per hole")
    return result


def environment_class(spec_path, capture=False):
    import math

    import torch
    from isaaclab.utils.math import matrix_from_quat, quat_apply
    from task import ResinPegEnv, tensor

    spec = json.loads(Path(spec_path).read_text())
    layout = assignments(spec)

    class OtherHolesEnv(ResinPegEnv):
        def __init__(self, cfg, **kwargs):
            if cfg.scene.num_envs != len(layout) or cfg.curriculum_stage != spec["curriculum_stage"]:
                raise ValueError("Environment layout/stage must match the locked experiment")
            super().__init__(cfg, **kwargs)
            self.layout = layout
            self.targets = torch.tensor(
                [spec["jobs"][a["job_index"]]["target_world_m"] for a in layout], device=self.device
            )
            self.fixed_banks = []
            for item in layout:
                row = spec["jobs"][item["job_index"]]["starts"][item["start_index"]]
                self.fixed_banks.append(
                    {
                        k: torch.tensor([row[k]], device=self.device, dtype=torch.float32)
                        for k in ["joint_pos", "peg_pos", "peg_quat_xyzw", "gripper_pos", "gripper_quat_xyzw"]
                    }
                )
            self.capture_terminal = False

        def _setup_scene(self):
            super()._setup_scene()
            self.observer_cameras = []
            if capture:
                import isaaclab.sim as sim_utils
                from isaaclab.sensors.camera import Camera, CameraCfg
                from isaaclab_physx.renderers import IsaacRtxRendererCfg

                for i in range(len(spec["jobs"])):
                    camera = Camera(
                        CameraCfg(
                            prim_path=f"/World/ComparisonCamera_{i}",
                            width=960,
                            height=540,
                            data_types=["rgb"],
                            renderer_cfg=IsaacRtxRendererCfg(),
                            spawn=sim_utils.PinholeCameraCfg(
                                focal_length=50, horizontal_aperture=36, clipping_range=(0.01, 20)
                            ),
                        )
                    )
                    # Global observers have one instance each. Never register
                    # them as per-environment sensors; reset/update them manually.
                    self.observer_cameras.append(camera)

        def _reset_idx(self, env_ids):
            if getattr(self, "capture_terminal", False):
                return
            if not hasattr(self, "fixed_banks"):
                return super()._reset_idx(env_ids)
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            previous = self.banks[self.curriculum_stage]
            try:
                for env_id in env_ids.tolist():
                    self.banks[self.curriculum_stage] = self.fixed_banks[env_id]
                    super()._reset_idx(torch.tensor([env_id], device=self.device, dtype=torch.long))
            finally:
                self.banks[self.curriculum_stage] = previous

        def metrics(self):
            pose = tensor(self.peg.data.root_pose_w)
            local = pose[:, :3] - self.scene.env_origins
            axis = quat_apply(pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
            tip = local + 0.075 * axis
            radial = (tip[:, :2] - self.targets[:, :2]).norm(dim=-1)
            tilt = torch.acos((-axis[:, 2]).clamp(-1, 1))
            depth = self.targets[:, 2] - tip[:, 2]
            samples = self.peg_samples.unsqueeze(0).expand(self.num_envs, -1, -1)
            quats = pose[:, None, 3:].expand(-1, samples.shape[1], -1)
            surface = local[:, None, :] + quat_apply(quats.reshape(-1, 4), samples.reshape(-1, 3)).reshape_as(samples)
            wall = chamfer_penetration(surface, spec, xp=torch).amax(dim=1)
            ee = self.ee_pose()
            expected = ee[:, :3] + quat_apply(ee[:, 3:], self.grasp_offset.expand(self.num_envs, -1))
            slip = (pose[:, :3] - expected).norm(dim=-1)
            invalid = (wall > 0.0002) | (surface[:, :, 2].amin(dim=1) < 0.7498) | (slip > 0.015)
            invalid |= ~torch.isfinite(pose).all(dim=-1)
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
            hole = self.scene.env_origins + self.targets
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

    return OtherHolesEnv
