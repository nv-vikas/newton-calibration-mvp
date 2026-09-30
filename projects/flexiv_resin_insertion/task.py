"""Cylindrical resin-peg task in actual Isaac Lab + Newton, not a kinematic animation.

This is a separate sim-only training experiment. No tuning-toolkit APIs or real
robot interfaces are changed. The peg is dynamic and is never welded/teleported
during an episode. Reset-time placement is explicit and excluded from success.
"""

import importlib.metadata
import json
import math
from functools import partial
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from geometry import block_penetration, cylinder_samples
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonShapeCfg
from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg, NewtonArticulationRootPropertiesCfg
from pxr import Gf, UsdGeom, UsdPhysics
from rewards import progress_reward

ARM_ARMATURE = 0.05


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


@configclass
class ResinPegCfg(DirectRLEnvCfg):
    decimation = 32
    episode_length_s = 24.0
    action_space = 6
    observation_space = 32
    state_space = 0
    prepared_dir: str = ""
    evaluation: bool = False
    capture: bool = False
    curriculum_stage: int = 0
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False)
    sim = SimulationCfg(
        dt=1 / 960,
        render_interval=32,
        use_newton_actuators=False,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                iterations=100,
                tolerance=1e-6,
                integrator="implicitfast",
                use_mujoco_contacts=False,
                njmax=2048,
                nconmax=1024,
            ),
            num_substeps=1,
            default_shape_cfg=NewtonShapeCfg(margin=0.00005, gap=0.00005),
            use_cuda_graph=True,
        ),
    )


class ResinPegEnv(DirectRLEnv):
    cfg: ResinPegCfg

    def __init__(self, cfg, **kwargs):
        self.prepared = json.loads((Path(cfg.prepared_dir) / "prepared.json").read_text())
        self.scene_manifest = self.prepared.get("scene_manifest", self.prepared["source_manifest"])
        self.curriculum_stage = cfg.curriculum_stage
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.episode_records = []
        super().__init__(cfg, **kwargs)
        self.arm_ids, _ = self.robot.find_joints([f"joint{i}" for i in range(1, 8)], preserve_order=True)
        self.grip_ids = [i for i, n in enumerate(self.robot.joint_names) if n not in [f"joint{j}" for j in range(1, 8)]]
        self.ee_id = self.robot.find_bodies("gripper_base")[0][0]
        if importlib.metadata.version("newton") != "1.2.1":
            raise RuntimeError("Recheck mass-matrix/armature convention before changing the pinned Newton runtime")
        # Newton 1.2.1 eval_mass_matrix is J^T M_body J: it excludes the rotor
        # armature that MJWarp DOES integrate. Without this diagonal the OSC
        # underestimates inertia and damping, especially at the wrist.
        self.armature_matrix = ARM_ARMATURE * torch.eye(7, device=self.device).unsqueeze(0)
        self.osc = OperationalSpaceController(
            OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                inertial_dynamics_decoupling=True,
                gravity_compensation=False,
                motion_stiffness_task=[300.0, 300.0, 300.0, 100.0, 100.0, 100.0],
                motion_damping_ratio_task=[1.0] * 6,
                nullspace_control="none",
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_potential = torch.zeros(self.num_envs, device=self.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.goal_pose[:, 6] = 1.0
        self.reset_q = torch.zeros(self.num_envs, 7, device=self.device)
        self.reset_ee_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.hold_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_stage = torch.full_like(self.hold_count, self.curriculum_stage)
        self.success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.invalid = torch.zeros_like(self.success)
        self.max_wall_violation = torch.zeros(self.num_envs, device=self.device)
        self.max_slip = torch.zeros(self.num_envs, device=self.device)
        self.peg_samples = torch.tensor(cylinder_samples(), device=self.device)
        self.script_depth = torch.full((self.num_envs,), -0.005, device=self.device)
        self.script_started = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.script_orientation = torch.zeros(self.num_envs, 4, device=self.device)
        self.grasp_offset = torch.tensor(
            self.scene_manifest["scene"]["peg_in_gripper"], device=self.device, dtype=torch.float32
        )[:3, 3]
        self.nominal_q = tensor(self.robot.data.default_joint_pos).clone()
        for i in self.grip_ids:
            name = self.robot.joint_names[i]
            # Close against the peg, with bounded torque; all six vendor joints
            # are explicitly driven (the archived asset has no runtime mimic).
            self.nominal_q[:, i] = 0.04 if "outer_finger" not in name else -0.04
        split = "heldout" if cfg.evaluation else "train"
        self.banks = {}
        for level in range(3):
            rows = [r for r in self.prepared["starts"] if r["stage"] == level and r["split"] == split]
            self.banks[level] = {
                key: torch.tensor([r[key] for r in rows], device=self.device, dtype=torch.float32)
                for key in ("joint_pos", "peg_pos", "peg_quat_xyzw", "gripper_pos", "gripper_quat_xyzw")
            }

    def _setup_scene(self):
        assets = Path(self.cfg.prepared_dir)
        initial = self.scene_manifest["scene"]["joint_positions_rad"]
        self.robot = Articulation(
            ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(assets / "Flexiv_Rizon4s_Grav.usd"),
                    rigid_props=MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                    articulation_props=NewtonArticulationRootPropertiesCfg(self_collision_enabled=True),
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.765), joint_pos=initial),
                actuators={
                    "arm": IdealPDActuatorCfg(
                        joint_names_expr=["joint[1-7]"],
                        stiffness=0.0,
                        damping=0.0,
                        effort_limit={"joint[1-2]": 123.0, "joint[3-4]": 64.0, "joint[5-7]": 39.0},
                        effort_limit_sim=1e9,
                        armature=ARM_ARMATURE,
                    ),
                    "gripper": IdealPDActuatorCfg(
                        joint_names_expr=["finger_joint", ".*_knuckle_joint", ".*_finger_joint"],
                        # Near critical damping for the declared 0.01 armature.
                        # Simulation initialization, NOT measured vendor gains.
                        stiffness=200.0,
                        damping=3.0,
                        effort_limit=5.0,
                        effort_limit_sim=1e9,
                        armature=0.01,
                    ),
                },
            )
        )
        self.peg = RigidObject(
            RigidObjectCfg(
                prim_path="/World/envs/env_.*/Peg",
                spawn=sim_utils.UsdFileCfg(usd_path=str(assets / "Peg_25.usd")),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.873), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        )
        # The exact static triangle mesh preserves the 25.5 mm bore.
        fixture = self.sim.stage.DefinePrim("/World/envs/env_0/HoleBlock", "Xform")
        fixture.GetReferences().AddReference(str(assets / "HoleBlock.usd"))
        UsdGeom.Xformable(fixture).AddTranslateOp().Set(Gf.Vec3d(0.15, 0.0, 0.75))
        table = UsdGeom.Cube.Define(self.sim.stage, "/World/envs/env_0/Table")
        table.CreateSizeAttr(1.0)
        table.AddTranslateOp().Set(Gf.Vec3d(-0.05, 0.0, 0.7225))
        table.AddScaleOp().Set(Gf.Vec3f(1.6, 0.9, 0.055))
        table.CreateDisplayColorAttr([(0.4, 0.45, 0.5)])
        UsdPhysics.CollisionAPI.Apply(table.GetPrim()).CreateCollisionEnabledAttr(True)
        root = UsdPhysics.FixedJoint(self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/root_joint"))
        # A world-anchored joint uses WORLD coordinates even when its prim is
        # inside env_0. Replication then applies each clone's relative offset.
        origin = self.scene.env_origins[0].cpu().tolist()
        root.CreateLocalPos0Attr(Gf.Vec3f(origin[0] - 0.45, origin[1], origin[2] + 0.765))
        # Verified USD chain: link7 --fixed--> flange --fixed--> gripper_base.
        # Their mounting overlap is intentional, not movable self-contact.
        # Newton 1.2.1 reads filteredPairs on collider prims, not body prims.
        wrist = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/link7/collisions/convex_mesh/mesh")
        gripper_mount = self.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/Grav_gripper/gripper_base/collisions")
        if not wrist.IsValid() or not gripper_mount.IsValid():
            raise RuntimeError("Fixed-mount collider paths changed; re-audit the collision filter")
        UsdPhysics.FilteredPairsAPI.Apply(wrist).CreateFilteredPairsRel().AddTarget(gripper_mount.GetPath())
        # The beta cloner defaults to convexifying EVERY collision mesh, which
        # fills the bore. Keep the authored meshes; never train on a solid hole.
        clone_physics = self.scene.cloner_cfg.physics_clone_fn
        if clone_physics is not None:
            self.scene.cloner_cfg.physics_clone_fn = partial(clone_physics, simplify_meshes=False)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["peg"] = self.peg
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.9, 1.0))
        light.func("/World/Light", light)
        if self.cfg.capture:
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab_physx.renderers import IsaacRtxRendererCfg

            # RTX is only the renderer; the physics backend remains Newton.
            self.camera = Camera(
                CameraCfg(
                    prim_path="/World/Camera",
                    height=720,
                    width=1280,
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=50.0, horizontal_aperture=36.0, clipping_range=(0.01, 20.0)
                    ),
                )
            )
            self.scene.sensors["overview"] = self.camera

    def ee_pose(self):
        return tensor(self.robot.data.body_link_pose_w)[:, self.ee_id].clone()

    def _pre_physics_step(self, actions):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1, 1)
        pose = self.ee_pose()
        self.goal_pose[:, :3] = pose[:, :3] + self.actions[:, :3] * torch.tensor(
            [0.001, 0.001, 0.002], device=self.device
        )
        rotation = self.actions[:, 3:] * 0.015
        angle = rotation.norm(dim=-1)
        axis = rotation / angle.clamp_min(1e-8).unsqueeze(-1)
        self.goal_pose[:, 3:] = quat_mul(quat_from_angle_axis(angle, axis), pose[:, 3:])
        self.osc.set_command(self.goal_pose)

    def _apply_action(self):
        pose = self.ee_pose()
        jacobian = tensor(self.robot.data.body_link_jacobian_w)[:, self.ee_id - 1, :, :][:, :, self.arm_ids]
        mass = tensor(self.robot.data.mass_matrix)[:, self.arm_ids, :][:, :, self.arm_ids]
        mass = mass + self.armature_matrix
        q = tensor(self.robot.data.joint_pos)[:, self.arm_ids]
        dq = tensor(self.robot.data.joint_vel)[:, self.arm_ids]
        reported_velocity = tensor(self.robot.data.body_link_vel_w)[:, self.ee_id]
        # Use velocity consistent with the numerically checked Jacobian and
        # actual joint feedback. Preserve the beta getter discrepancy for audit.
        velocity = (jacobian @ dq.unsqueeze(-1)).squeeze(-1)
        self.velocity_consistency_error = (reported_velocity - velocity).abs().amax(dim=1)
        tau = self.osc.compute(
            jacobian_b=jacobian,
            current_ee_pose_b=pose,
            current_ee_vel_b=velocity,
            mass_matrix=mass,
            current_joint_pos=q,
            current_joint_vel=dq,
            nullspace_joint_pos_target=self.reset_q,
        )
        self.robot.set_joint_effort_target_index(target=tau, joint_ids=self.arm_ids)
        self.commanded_tau = tau.clone()
        self.robot.set_joint_position_target_index(target=self.nominal_q[:, self.grip_ids], joint_ids=self.grip_ids)

    def metrics(self):
        pose = tensor(self.peg.data.root_pose_w)
        local_pos = pose[:, :3] - self.scene.env_origins
        axis = quat_apply(pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
        tip = local_pos + 0.075 * axis
        center = torch.tensor([0.15, 0.0], device=self.device)
        radial = (tip[:, :2] - center).norm(dim=-1)
        tilt = torch.acos((-axis[:, 2]).clamp(-1, 1))
        depth = 0.79 - tip[:, 2]
        samples = self.peg_samples.unsqueeze(0).expand(self.num_envs, -1, -1)
        sample_quats = pose[:, None, 3:].expand(-1, samples.shape[1], -1)
        surface = local_pos[:, None, :] + quat_apply(sample_quats.reshape(-1, 4), samples.reshape(-1, 3)).reshape_as(
            samples
        )
        wall_violation = block_penetration(surface, xp=torch).amax(dim=1)
        ee = self.ee_pose()
        expected = ee[:, :3] + quat_apply(ee[:, 3:], self.grasp_offset.expand(self.num_envs, -1))
        slip = (pose[:, :3] - expected).norm(dim=-1)
        invalid = (wall_violation > 0.0002) | (surface[:, :, 2].amin(dim=1) < 0.7498) | (slip > 0.015)
        invalid |= ~torch.isfinite(pose).all(dim=-1)
        seated = (depth >= 0.035) & (radial <= 0.00035) & (tilt <= math.radians(1.0)) & ~invalid
        return {
            "radial": radial,
            "tilt": tilt,
            "depth": depth,
            "slip": slip,
            "wall_violation": wall_violation,
            "invalid": invalid,
            "seated": seated,
        }

    def _get_dones(self):
        self.m = self.metrics()
        self.hold_count = torch.where(self.m["seated"], self.hold_count + 1, 0)
        self.max_wall_violation = torch.maximum(self.max_wall_violation, self.m["wall_violation"])
        self.max_slip = torch.maximum(self.max_slip, self.m["slip"])
        self.invalid |= self.m["invalid"]
        self.success = (self.hold_count >= math.ceil(0.25 / self.step_dt)) & ~self.invalid
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.success | self.invalid, time_out

    def potential(self, m):
        aligned = torch.exp(-m["radial"] / 0.003) * torch.exp(-m["tilt"] / 0.1)
        depth_reward = (m["depth"] / 0.035).clamp(0, 1)
        return 2.0 * aligned + 4.0 * depth_reward * aligned

    def _get_rewards(self):
        potential = self.potential(self.m)
        reward = progress_reward(self.previous_potential, potential, self.success, self.invalid)
        reward -= 0.005 * (self.actions - self.previous_actions).square().sum(-1)
        self.previous_potential.copy_(potential)
        return reward

    def _get_observations(self):
        ee = self.ee_pose()
        hole = self.scene.env_origins + torch.tensor([0.15, 0.0, 0.79], device=self.device)
        rotation = matrix_from_quat(ee[:, 3:])[:, :, :2].reshape(self.num_envs, 6)
        # No true peg state in actor observations. Hole pose is known in this MVP.
        obs = torch.cat(
            [
                tensor(self.robot.data.joint_pos)[:, self.arm_ids],
                tensor(self.robot.data.joint_vel)[:, self.arm_ids],
                (ee[:, :3] - hole),
                rotation,
                self.actions,
                tensor(self.robot.data.joint_pos)[:, self.grip_ids[:3]],
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not hasattr(self, "banks"):
            return super()._reset_idx(env_ids)
        active = self.episode_length_buf[env_ids] > 0
        for idx in env_ids[active].tolist():
            self.episode_records.append(
                {
                    "env_id": idx,
                    "success": bool(self.success[idx]),
                    "invalid": bool(self.invalid[idx]),
                    "max_slip_m": float(self.max_slip[idx]),
                    "max_wall_violation_m": float(self.max_wall_violation[idx]),
                    "stage": int(self.episode_stage[idx]),
                    "terminal_peg_pose": tensor(self.peg.data.root_pose_w)[idx].cpu().tolist(),
                }
            )
        self.completed_episodes += int(active.sum())
        self.successful_episodes += int(self.success[env_ids][active].sum())
        if bool(active.any()):
            self.extras.setdefault("log", {})["Metrics/terminal_success_rate"] = (
                self.success[env_ids][active].float().mean()
            )
            self.extras["log"]["Metrics/invalid_rate"] = self.invalid[env_ids][active].float().mean()
        super()._reset_idx(env_ids)
        bank = self.banks[self.curriculum_stage]
        self.episode_stage[env_ids] = self.curriculum_stage
        select = torch.randint(len(bank["joint_pos"]), (len(env_ids),), device=self.device)
        q = self.nominal_q[env_ids].clone()
        q[:, self.arm_ids] = bank["joint_pos"][select]
        # Start at exact geometric 25 mm aperture; drives close during physics.
        original = self.scene_manifest["scene"]["joint_positions_rad"]
        for i in self.grip_ids:
            q[:, i] = original[self.robot.joint_names[i]]
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        self.reset_q[env_ids] = q[:, self.arm_ids]
        self.reset_ee_pose[env_ids] = torch.cat(
            [bank["gripper_pos"][select] + self.scene.env_origins[env_ids], bank["gripper_quat_xyzw"][select]], dim=-1
        )
        peg_pose = torch.cat(
            [bank["peg_pos"][select] + self.scene.env_origins[env_ids], bank["peg_quat_xyzw"][select]], dim=-1
        )
        self.peg.write_root_pose_to_sim_index(root_pose=peg_pose, env_ids=env_ids)
        self.peg.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        self.hold_count[env_ids] = 0
        self.success[env_ids] = False
        self.invalid[env_ids] = False
        self.max_wall_violation[env_ids] = 0
        self.max_slip[env_ids] = 0
        self.script_depth[env_ids] = -0.005
        self.script_started[env_ids] = False
        self.actions[env_ids] = 0
        self.previous_actions[env_ids] = 0
        self.previous_potential[env_ids] = self.potential(self.metrics())[env_ids]
        # The installed OSC has a stateless reset() with no per-env argument.
        self.osc.reset()

    def scripted_action(self, insert=True):
        """Diagnostic only: known geometry, not a trained policy or real command."""
        pose = tensor(self.peg.data.root_pose_w)
        axis = quat_apply(pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
        tip = pose[:, :3] + 0.075 * axis - self.scene.env_origins
        action = torch.zeros(self.num_envs, 6, device=self.device)
        ee = self.ee_pose()
        position_error, rotation_error = compute_pose_error(
            ee[:, :3], ee[:, 3:], self.reset_ee_pose[:, :3], self.reset_ee_pose[:, 3:], rot_error_type="axis_angle"
        )
        action[:, 3:] = rotation_error / 0.015
        if insert:
            desired_peg_quat = torch.zeros(self.num_envs, 4, device=self.device)
            desired_peg_quat[:, 0] = 1.0
            # Freeze the orientation target after grip settling. Repeatedly
            # integrating peg orientation error when contact jams the peg can
            # keep rotating the wrist inside its grasp instead of recovering.
            first = ~self.script_started
            correction = quat_mul(desired_peg_quat, quat_conjugate(pose[:, 3:]))
            self.script_orientation[first] = quat_mul(correction, ee[:, 3:])[first]
            self.script_started[:] = True
            _, wrist_rotation_error = compute_pose_error(
                ee[:, :3], ee[:, 3:], ee[:, :3], self.script_orientation, rot_error_type="axis_angle"
            )
            action[:, 3:] = wrist_rotation_error / 0.015
            center = torch.tensor([0.15, 0.0], device=self.device)
            midline = tip[:, :2] - axis[:, :2] * 0.0185
            action[:, :2] = (center - midline) / 0.001
            # Predict clearance at BOTH ends of the final inserted section,
            # using the 0.25 mm radial gap with a 0.05 mm reserve.
            far_end = tip[:, :2] - axis[:, :2] * 0.037
            ready = torch.maximum((tip[:, :2] - center).norm(dim=-1), (far_end - center).norm(dim=-1)) < 0.0002
            # Hold an ABSOLUTE height while aligning. A zero relative command
            # reanchors to the sagging pose and cannot hold an external payload.
            tracking = (self.script_depth - (0.79 - tip[:, 2])).abs() < 0.001
            self.script_depth = torch.where(
                ready & tracking, (self.script_depth + 0.00015).clamp_max(0.037), self.script_depth
            )
            action[:, 2] = (0.79 - self.script_depth - tip[:, 2]) / 0.002
        else:
            action[:, :3] = position_error / torch.tensor([0.001, 0.001, 0.002], device=self.device)
        return action.clamp(-1, 1)
