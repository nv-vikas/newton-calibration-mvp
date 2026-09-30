# Flexiv reference binding for the qualification product feature

Core documentation: [precision insertion qualification](../../../docs/precision_insertion_qualification.md).
The source task/checkpoint stays frozen. This directory contains the scene-specific
adapter; orchestration, gates and durability live in `newton_calibration.qualification`.

## Docker layout

A thin product image can be built without downloading or installing packages:

```bash
docker build --network=none --pull=false \
  -f projects/flexiv_resin_insertion/qualification/Dockerfile \
  -t newton-qualification:preview .
```

Its entry point is `newton-calibration qualify` through the bundled Isaac Lab
Python runtime. Record the resolved base and output image IDs with deployment;
the local base tag alone is not an immutable release identity.

Use the already verified `flexiv-resin-insertion:20260916-v24` image (resolved image
SHA recorded with the run). Mount the product source as `/feature:ro`, prepared
USD assets as `/work/prepared:ro`, original run as `/work/output:ro` and a **new**
result directory as `/results`. Set `PYTHONPATH=/feature/src:/feature`. Never replace
`/work/flexiv_resin_insertion`, the frozen source inside that image.

```bash
/workspace/isaaclab/_isaac_sim/python.sh \
  -m projects.flexiv_resin_insertion.qualification.bootstrap \
  --output /results --prepared /work/prepared --training-run /work/output \
  --checkpoint /work/output/checkpoints/model_731.pt
```

Bootstrap solves starts 0,1,3,4 from the original source bank at the 25.05 mm hole.
Starts 0,1 are development; 3,4 are reserved. These are different from starts
14,7,2,12 used in the previous comparison. They are not new independent real trials.
Other consumers should predeclare their own split before examining outcomes.

Use `--geometry-only` for a first smoke check. To continue the same pinned job:

```bash
export NEWTON_QUALIFICATION_BINDING=/results/binding.json
/workspace/isaaclab/_isaac_sim/python.sh -m newton_calibration.cli qualify run \
  --job /results/job \
  --backend-factory projects.flexiv_resin_insertion.qualification.factory:create_backend
```

The policy stays at 30 Hz. Contact margin/gap bind explicitly to peg and fixture
colliders before Newton model finalization; robot/gripper/table collider settings
are untouched. Every experiment records the originally imported and applied
values. The comparison uses the declared recipe baseline, which should not be
assumed identical to imported USD defaults without checking that record.
Only numerical settings change; no friction, dimensions,
controller gains, actor weights or action scaling are retuned. The controlled
diagnostic uses true simulated peg pose and known geometry; it is not a deployable
perception policy. No real Flexiv API is connected.

Every source file, prepared asset, policy/config and trial bank is pinned. An
adapter bug fix requires a new immutable source snapshot and new job baseline;
never copy new code into an active snapshot and continue under its old identity.

Video is on by default for controlled insertion and both policy conditions. The
centered free-peg/rim probes are numeric diagnostics with full traces, not presentation
clips. A stage can be blocked legitimately; do not bypass it to produce a success video.
