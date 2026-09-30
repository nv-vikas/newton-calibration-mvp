# Flexiv resin insertion — Horde run, September 15–16, 2026

> Historical snapshot, preserved when publishing pending work on September 30,
> 2026. Statements below about a running job describe September 15–16, not live
> Horde status. This publication did not restart training, inspect the node or
> establish a new policy, calibration or real-transfer result.

## What is actually established

- Actual Isaac Lab **3.0.0-beta2**, **Newton 1.2.1 / MuJoCo Warp** physics.
- The dynamic 25 mm peg is held by simulated finger contact, not a weld.
- Independent drop tests distinguish the open bore from the solid rim.
- Scripted qualification passed in **4/4 cloned environments**: at least
  **35 mm insertion**, held for **0.25 seconds**, without a flagged wall/table
  violation or dropped peg. This establishes nominal feasibility, not policy
  performance or real-robot safety. The sampled geometry screen is not a
  contact-force measurement or exact penetration certificate.
- Initial hold slip was at most **1.29 mm**; relative grasp displacement during
  the complete scripted insertion reached **7.04 mm**. The grasp is therefore
  not rigid or perfectly stable, even though it passed the declared 15 mm
  retained-grasp limit. Do not describe this as a calibrated gripper.
- PPO completed **2 updates / 8,192 transitions** in 32 environments and wrote
  `model_1.pt`. A fresh Python process loaded all **76 checkpoint tensors** and
  verified finite values. Actor, critic and optimizer states are present.
- The detached supervisor then passed its own fresh qualification, restored
  that checkpoint, and completed learning iteration 2 (the **third total PPO
  update**). The first resumed update took 27.33 seconds and reported finite
  losses. **Resume is verified; training is running**, not merely queued.
- A checkpoint and qualification reports were copied off-node into this repo's
  ignored `output/resin_insertion/run_v24/` directory.
- **No trained-policy held-out result exists yet.** Early smoke-run rate fields
  must not be treated as evaluation; two updates are shorter than a full episode
  for a non-terminating environment.

## Current job

Detached container: `resin-train-20260916-v24`.

It has requalified the same immutable image, loaded the smoke-test checkpoint,
and is training toward 1,000 total PPO updates. The supervisor has a **48-hour total
budget**, at most two checkpoint-based retries, and a final 200-trial simulated
held-out evaluation. Those evaluation trials reuse 16 held-out initial poses;
they are not 200 independent real-world trials.

Image: `flexiv-resin-insertion:20260916-v24`

Image ID:
`sha256:c265895d8ada94690a81bb7b03c758fe1a56060712920d14c1567ad5254e22ae`

Source/asset fingerprint:
`6f3c8af74dc9ba86be354675e5d050bef7eaf78a842ab6301f5b5e6023b42be2`

Host output:
`/home/horde/workspace/resin_insertion_20260916/run_v24`

Host prepared assets:
`/home/horde/workspace/resin_insertion_20260916/prepared_v3`

The four-environment qualification is archived remotely at
`run_v24/qualification_4env/qualification.json`; the supervisor writes a fresh
qualification at the run root. Consult `status.json`, `train_0.log`,
`progress.jsonl`, `latest_checkpoint.json`, and eventually
`heldout_evaluation.json` for current state. Docker exit code alone is not proof
of success: Isaac shutdown previously masked Python exceptions.

## Persistence and limits

The laptop may disconnect once the detached job is running. The Horde allocation
must remain alive; this setup does not reprovision a reclaimed node. Checkpoints
restore policy and optimizer but restart episodes, not the exact physics state.
The local copy is a backup of the early checkpoint, not continuous replication.

The initial measured throughput was approximately 124 transitions/second. That
would put 1,000 updates near nine hours if sustained, **not** a prediction that
an effective policy will be learned in nine hours. Curriculum advancement and
held-out evaluation determine what has actually been learned.

## Scope caveats

This is an isolated **simulation baseline**, not calibration or deployment.
Resin density/friction are unmeasured priors. Control is Isaac Lab OSC, not the
Flexiv real deployment controller. The peg starts grasped near a known hole;
there is no pickup or visual perception policy. Original assets, the MVP1
toolkit, and earlier experimental artifacts were preserved.

Code validation: 279 existing toolkit tests and 18 isolated experiment tests
passed; experiment lint passed. Physics qualification is separate from those
unit tests. The earlier v12 viewport recording is a failed historical diagnostic,
not a video of this qualified run or a trained policy.
