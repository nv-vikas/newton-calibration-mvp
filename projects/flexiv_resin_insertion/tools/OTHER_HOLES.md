# Other-hole frozen-policy comparison

This is an isolated simulation/video experiment. It does not retrain the policy,
change Newton parameters, edit the source USDs, or modify the original training
job. It does not operate a real robot.

## Predeclared design

- Checkpoint: `model_731.pt`, 732 updates, SHA-256
  `b2ccab2b5e03a4a61371c94021d06ff0408f607e59fbef6977ee76085c578a90`.
- Runtime: the frozen `flexiv-resin-insertion:20260916-v24` image, Isaac Lab
  3.0.0-beta2 / Newton 1.2.1 / MuJoCo Warp.
- Four targets: center 25.50 mm without chamfer, and outer 25.50 mm, 25.25 mm,
  25.05 mm holes with their authored 2 mm chamfers.
- Four matched stage-1 held-out initial offsets per target: source-bank indices
  **14, 7, 2, 12**, in that order. The arm's initial joint configuration is solved
  from the USD for each target; the target is supplied through the existing
  observation fields. Actor weights, action interface and physics stay fixed.
- The recorded start is **index 14 for every hole**, chosen before observing
  this comparison's results. It was the starting condition shown in the earlier
  25.75 mm video. No replacement by a better-looking outcome is allowed.
- Sixteen separate simulated episodes run in cloned environments. Four global
  cameras observe the preselected episodes and are reset/updated manually,
  outside the per-environment sensor-reset loop.

## Geometry and interpretation

`prepare_other_holes.py` verifies the bottom bore, throat and mouth vertices in
the original triangle-mesh collider. It solves all required poses with bounded
multi-start IK and refuses to continue if any pose misses the tolerance.

`run_other_holes.py` first records free-peg center-bore and solid-rim diagnostics.
A failed diagnostic does not trigger hidden retuning or a policy retry. The
exploratory policy attempt is still shown, with its qualification limitation
recorded separately.

Original task completion checks are retained: at least 35 mm insertion depth,
at most 0.35 mm radial error, at most 1 degree tilt, sustained for 0.25 s, and no
flagged invalid state. The original sampled-wall invalidity threshold is 0.2 mm.

An additional, predeclared diagnostic reports whether a task pass also has
passing bore/rim probes and maximum sampled wall penetration below the smaller
of 0.2 mm or half the nominal radial clearance. This is a conservative reporting
filter, **not a certification** of exact nonpenetration, force accuracy or safety.
It does not change actions, contacts, termination or success thresholds.

This matters especially for 25.05 mm: nominal radial clearance is only 25 um,
below the unchanged 50 um shape margin/gap settings and 200 um invalidity gate.
The 25.25 mm hole's 125 um clearance is also below that original invalidity gate.
The actual collider remains the supplied faceted triangle mesh; the sampled
analytic screen is only an approximation.

## Artifacts

- `spec.json`: verified geometry, all 16 starting poses and predeclared scope.
- `fixture_probes.json`: actual Newton free-peg diagnostics.
- `outcomes_so_far.json`: durable terminal outcomes during execution.
- `result.json`, `trajectories.json`, `status.json`: complete outcomes, applied
  (clipped) actions, provenance hashes, video selection and status.
- `center_2550.mp4`, `chamfer_2550.mp4`, `chamfer_2525.mp4`, `chamfer_2505.mp4`:
  preselected learned-policy episodes, including failures.
- `other_holes_comparison.mp4`: synchronized 2x2 presentation. Finished panels
  explicitly hold their terminal frame; all clips include a 2-second end hold.

These are four matched simulated starts per hole, not a dependable success-rate
estimate. Hole location and initial joint configuration also change, so the
comparison does not isolate diameter or chamfer as the cause of any difference.
Material properties remain unmeasured assumptions, not calibrated values.

## Completed result — September 16, 2026

The 16-episode `other_holes_policy731_v1` run completed: center 25.50 mm **2/4**,
chamfered 25.50 mm **3/4**, chamfered 25.25 mm **2/4**, chamfered 25.05 mm **0/4**.
Task and supplemental-screen counts agreed. The selected videos show successes
for the first three holes and the failed 25.05 mm attempt. All nine unsuccessful
trials exceeded the relative grasp-displacement limit; no trial was discarded.

The smallest hole also failed its independent centered free-peg probe, stopping
near 6.5 mm depth. Do not interpret its policy result independently of the frozen
runtime's precision-fit limitations. No tuning or retraining was done.

Complete local evidence and videos:
`output/resin_insertion/other_holes_policy731_v1/RESULT_SUMMARY.md` and its
`output/` subdirectory. All five videos decoded cleanly, and the initial and
terminal viewport frames were visually checked.
