# Controller discovery in MVP1

The task says **what** the robot should do; the controller profile records
**how commands reach the robot**. A USD does not identify that control path.
This feature inspects an existing Isaac Lab config and asks the user to confirm
the simulation controller and the real interface separately. No robot motion,
policy training, Cartesian fitting, or MVP2/MVP3 implementation is added.

## User journey

1. Supply the calibration environment plus the active Isaac Lab task config.
2. Discover its action family, implementation, actuator settings and command rate.
   Multiple action terms require selection; the toolkit does not guess which is
   the arm controller. Gripper configuration remains in the captured settings.
3. Review the detected simulation settings. Supply missing frame details, then
   the real driver, exact control mode, command semantics, frame, units, rate
   and configuration source. Record payload, smoothing, limits and source-code
   revision under `settings` where relevant. Never put credentials here.
4. Confirm both sides. This confirms configuration, **not** response equivalence.
5. Run analyze/plan. Joint-PD proposals can still be generated without real
   evidence, with preview enabled by default. Unsupported controllers cannot be
   substituted with joint-PD fitting or motion files.

The controller fingerprint locks the declared configuration, not executable
source bytes. Include a source revision when reproducibility requires it.
Existing joint-PD baseline confirmation and bounds checks still apply; profile
settings are not automatically copied into runtime gains.

## Python / agent interface

`env` below is the existing `ArticulationEnvCfg` calibration surface, not the
full training task. `lab_env` is the actual Isaac Lab environment (or its already
constructed config). No new physics application needs to start for discovery.

```python
from newton_calibration.isaaclab import tuning, discover_controller

profile = discover_controller(lab_env, action_name="arm_action")
# profile.simulation contains the discovered data; profile.real is empty.
# Save/review this data; never set confirmation flags on behalf of the user.

analysis = tuning.analyze(
    env=env,
    isaaclab_env=lab_env,
    controller_action="arm_action",
    evidence=None,
    workdir="runs",
)
questions = analysis.controller["questions"]
```

An external agent presents the stable question IDs and writes the user's
answers into a `ControllerProfile`. Pass it as `controller_profile=profile`
to `analyze` or `assist`, or set `ArticulationEnvCfg.controller_profile` to its
dictionary. A surface may expose `describe_controller_config()` so analyze
discovers it automatically. Existing five-call signatures remain usable.

When both a live config and a saved profile are supplied, a changed controller,
gain, action scale, actuator configuration or timing rejects the old profile.
Rediscover and confirm a new revision. The operator-specified frame is retained
because discovery deliberately does not guess it.

## CLI setup and prompts

A trusted local factory returns the configured task, **not** a file path or
an executable from a profile. Importing that factory executes trusted Python;
only use your own reviewed setup code. The data-only JSON loader never imports
the `implementation` string.

```bash
newton-calibration controller \
  --env-factory my_scene:make_task_cfg \
  --action-name arm_action \
  --name flexiv-insertion-v1 \
  --interactive \
  --output profiles/flexiv-insertion-v1.json
```

`--interactive` requires a terminal. Without it, the command saves the discovered
profile and prints machine-readable missing-information questions. With no
factory, it creates a blank profile and asks the user to specify the controller;
it cannot infer a controller from the word “insertion.” `--profile` reviews an
existing JSON, and `--output` must name a new file, never an overwrite.

```bash
newton-calibration assist --config collection-job.json \
  --controller-profile profiles/flexiv-insertion-v1.json \
  --controller-env-factory my_scene:make_task_cfg \
  --controller-action arm_action
```

Controller discovery is a setup operation, not a sixth calibration lifecycle
call. The `controller` command exits 2 when details or adapter support are
missing, even though it saved a useful profile. `assist` also exits 2 when
controller support blocks collection.

## Support boundaries

| Detected controller | MVP1 behavior |
|---|---|
| Absolute joint-position PD | Existing joint-replay fitting and provisional collection, subject to readiness checks |
| Joint-position action with identified IdealPD actuators | Discovery; explicitly declare `simulation.settings.evidence_boundary=postprocessed_joint_targets` only if evidence and the scene adapter actually use downstream absolute joint targets |
| Joint-position action with other/unverified actuators | Discovery only; a position action does not prove its actuator model is supported |
| Relative joint position, joint velocity or joint effort | Discovery only; no supported calibration adapter |
| Cartesian OSC or IK | Discovery only; no joint-PD substitution, no Cartesian calibration or collection |
| Multiple/unknown action terms | Ask for selection/configuration; do not guess |
| Legacy MVP1 input without a structured profile | Preserve the existing workflow and fingerprint format; report that deployment-controller discovery has not occurred |

Detected action wrappers are not equivalent to the joint setpoints passed to
the replay runtime. Merely labelling a profile does not implement a command
conversion. Raw policy actions, mismatched units/frames/rates and Cartesian
commands must not be passed to the joint-position runner. A supported profile
requires radians in joint coordinates. Motion CSV rates must match its declared
command rate; simulation integration dt is a separate quantity.

## Evidence and outputs

New profiles use `ArticulationEnvCfg` (including for SO-101) and require an explicit controller binding on real tabular evidence:

```python
evidence = bind_evidence_files(
    # existing file, signal, coordinate and train/heldout arguments ...
    controller_profile_fingerprint=profile.fingerprint,
)
```

Only bind a fingerprint after checking which configuration produced the logs.
Missing or mismatched capture metadata blocks fitting; do not “fix” a mismatch
by relabelling old logs. Legacy evidence without a profile remains supported.

Each analysis writes `controller_discovery.json` and includes the profile,
fingerprint, questions and evidence-binding status in `analysis.json`.
Collection writes `CONTROLLER_REVIEW.md` and includes the discovery in
`agent_assistance.json` and the hashed command plan. Fitting plans and packages
carry the environment's complete profile; package loading checks it against the
robot profile and evidence. Changing it after analysis requires a new run.

For an insertion/OSC controller, expect `controller_action_required`, zero
joint-PD motion files, and a blocked preview—not a fabricated calibration result.
For supported provisional joint motion, missing real confirmation never grants
hardware execution approval.

## Verification scope

CPU contract tests exercise discovery with Isaac Lab-shaped config doubles,
downstream subclasses, ambiguity, unit/rate mismatches, stale confirmations,
evidence binding, CLI questions and package round-trip/tamper rejection.
These are not a live Cartesian-controller integration test or real calibration.
