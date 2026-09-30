# Flexiv resin peg insertion — isolated simulation experiment

This project does **not** modify the five-call calibration toolkit, execute real
robot commands, or claim sim-to-real transfer. The user authorized a new sim-only
insertion baseline after the earlier MVP2/3 work was paused; those earlier
artifacts remain untouched.

## Scope and assumptions

- Flexiv Rizon4s + Grav; source robot and original STL-derived USD geometry from
  `../flexiv_mvp1/flexiv_peg_scene/assets`.
- 25 mm diameter, 75 mm long peg; 25.5 mm bore in a fixed 40 mm block.
- Peg starts grasped, above the hole. There is no pickup policy, welded peg,
  episode-time teleportation, camera-based perception, or force-sensor policy.
- The peg is held 12 mm farther out than in the earlier scene. Measured from
  the USD, the original grasp exposed only 33.9 mm, less than the required
  35 mm insertion depth. The revised grasp exposes 45.9 mm; preparation checks
  at least 5 mm fingertip clearance beyond the required depth. Original assets
  and their manifest are preserved alongside the new scene manifest.
- Peg and block are modeled as **rigid resin**, density 1200 kg/m³, static/dynamic
  friction 0.5/0.4, restitution zero. These are unmeasured priors, **not material
  measurements or a calibrated resin model**. Soft resin requires reevaluation.
- The preserved vendor gripper was adapted in the earlier scene to drive all six
  finger joints explicitly and includes small-link mass priors. This is not a
  verified model of the real Grav mechanism.
- Isaac Lab Cartesian OSC drives arm torques. Body-level ideal gravity
  compensation is enabled. This is not Flexiv Elements Studio's controller.
- Explicit arm/gripper control updates at 960 Hz; the policy acts at 30 Hz.
- The OSC inertia matrix includes the configured 0.05 kg·m² joint-armature
  diagonal. In the pinned Newton 1.2.1 implementation, `eval_mass_matrix`
  computes only `Jᵀ M_body J`, while MJWarp also integrates armature. The task
  refuses another Newton version until this convention is rechecked, to avoid
  accidentally double-counting rotor inertia after a runtime upgrade.
- The beta Newton cloner's blanket convexification is disabled for this scene:
  converting the concave fixture into a convex hull closes its bore. Qualification
  independently drops the peg through the bore and onto the solid rim before
  testing the robot. The tabletop collider is explicitly enabled.
- Only the intentional link7–gripper-base mounting overlap is collision-filtered;
  the USD connects those parts through two fixed joints. Other self-collisions
  remain enabled. Filters are authored on collider prims for Newton 1.2.1.
- OSC damping uses `J * joint_velocity`, consistent with the numerically checked
  world-frame Jacobian. The beta link-velocity getter discrepancy is recorded
  separately; this workaround is scoped to this pinned experiment.
- The reference repository supplied the design pattern (OSC, curriculum, PPO);
  this is a new cylindrical-peg task, not its DisplayPort task or checkpoint.

## Sequence

1. `prepare.py`: copy to new resin assets, preserve original assets, solve 192
   start poses from the USD kinematic chain. Separate train/held-out pose banks.
2. `run.py --mode qualify`: bore/rim drop probes, runtime Jacobian check, then
   actual Newton hold and scripted insertion. Long
   training is prohibited unless this passes with the same source/asset hash.
3. `run.py --mode train`: PPO, 32 environments initially, maximum 1000 iterations
   or 48 hours per attempt, checkpoint every 10 iterations. Curriculum advances
   only on observed training success, not an assumed elapsed schedule.
4. `run.py --mode evaluate`: 200 sim trials at the hardest start-pose level using
   the held-out bank. Report successes AND invalid-contact/drop outcomes. There
   are 16 unique hardest-level held-out starts, so repeated trials are explicitly
   reported and must not be described as 200 independent real trials.

Success requires insertion depth ≥35 mm, alignment, retained grasp, a sustained
seat for 0.25 s, and no detected wall/table violation. Geometry checks are an
additional anti-exploit screen, not physical force validation or proof that every
possible collision issue has been excluded.

The wall screen samples the peg's surface and caps against the known bored
block. It measures depth into solid material, including the nearest top/bottom
surface; it does not confuse horizontal rim misalignment with penetration.
This is still a sampled geometric check, not an exact mesh-distance certificate.

Scripted qualification can inspect true simulated peg pose to diagnose feasibility.
The learned actor does **not** receive that privileged peg pose. Its observations
are robot/gripper feedback, gripper pose relative to the known hole, and prior
actions. Scripted success therefore does not demonstrate learned-policy success.

PPO uses discounted potential-difference shaping plus a terminal seating bonus,
invalid-contact penalty and time/action-change costs. Holding an almost-inserted
pose does not earn a repeated positive depth reward. The pinned RSL-RL adapter
removes only Isaac Lab's four deprecated pre-v5 model configuration fields;
the actor's explicit Gaussian distribution remains unchanged.

## Runtime and persistence

Base image: `flexiv-peg-scene:agent-assist` (Isaac Lab 3.0.0-beta2,
Newton 1.2.1, Torch 2.10.0+cu128, RSL-RL 5.0.1). The Python extension package
version is not the Isaac Lab release number.

Run in a detached Docker container with `/home/horde/workspace` bind-mounted.
Create output directories on the host as the Horde user **before** using a bind
mount. The image runs as non-root `isaaclab`; Docker's auto-created root-owned
directories are not writable. Prefer `--mount` (which rejects missing sources)
over silently auto-creating bind paths. `run.py` tests output writes before Kit
starts and records import failures separately.
Laptop/VPN disconnection does not stop that container. The node itself must
remain allocated. The inspected Horde session has a bound persistent workspace;
the returned metadata does not specify a lease expiry, and uninterrupted service
is not guaranteed. A checkpoint on the same server is not an off-node backup.

Resume restores policy and optimizer. Episodes restart; this is not bitwise
continuation of the physics state. Changed source/asset fingerprints reject old
qualifications and checkpoints. No automatic controller/physics changes are
made by the training loop.

`supervise.py` runs qualification → training → evaluation with a global budget
and at most two checkpoint-based retries. An old success record cannot authorize
training after a failed or stale qualification. `--capture` on `run.py` requests
an actual RTX viewport recording during qualification (not an animation).

## Commands inside the prepared container

```sh
/workspace/isaaclab/isaaclab.sh -p /work/flexiv_resin_insertion/run.py \
  --mode qualify --prepared /work/prepared --output /work/run --num_envs 1 --headless

/workspace/isaaclab/isaaclab.sh -p /work/flexiv_resin_insertion/run.py \
  --mode train --prepared /work/prepared --output /work/run --num_envs 32 \
  --iterations 1000 --max_wall_hours 48 --resume --headless
```

Do not label an initialized scene, a scripted controller, a random policy or an
unvalidated checkpoint as a trained insertion policy. Consult `status.json`,
`qualification.json`, `progress.jsonl`, and `heldout_evaluation.json`.

## Learned-policy progress video

`tools/render_policy.py` runs the checkpoint's PPO actor with deterministic
inference in the **frozen training image**. Mount this helper separately, outside
the fingerprinted training package, and mount the training run and prepared
assets read-only. Use a fresh writable video output directory. The helper checks
the source/asset fingerprint against both the checkpoint and qualification.

It records one predetermined held-out start at the checkpoint's curriculum
stage, with no success-based trial selection and no scripted-controller calls.
The video labels the update count, insertion depth, grasp displacement and
outcome. Automatic reset is suppressed only at the terminal frame; no physics
steps are taken beyond termination. The final two seconds hold that frame.
`rollout.json` records the checkpoint hash, seed, actions, poses and outcome.
An early checkpoint video is progress evidence, not a validated insertion policy
or real-world transfer result.
