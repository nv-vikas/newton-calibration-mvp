# Precision insertion qualification — developer preview

An agent-executable, four-stage **simulation qualification** workflow. It diagnoses
whether a tight insertion can be represented and controlled in Isaac Lab/Newton,
then evaluates a **frozen** learned policy. It does not train a policy, operate
hardware, fit unmeasured material properties, or claim real-world calibration.

This is a sibling of the existing five-call real-evidence calibration workflow.
It deliberately does not disguise simulation-only tests as `fit(real_evidence)`.

## The user experience

Supply a supported scene binding, exact policy checkpoint, a data-only recipe and
separate development/validation starting conditions. Run through Python or CLI.
An optional agent uses exactly the same API and receives the next permitted stage,
measurements, blocked reason, trial records and artifact paths.

| Stage | Work | Gate and output |
|---|---|---|
| Geometry | Inspect actual collision geometry, not the image | Full-depth supported-family coverage, clearance, geometry error and unobstructed bore |
| Solver | Baseline, bounded declared sweep or installed optimizer; repeat and refined-physics checks | Valid bore and rim probes; nonzero replay/convergence tolerances; first qualified configuration |
| Controlled insertion | Hold, slow free motion and scripted insertion with a physical grasp | Phase completion, depth, sustained hold, conservative penetration and grasp-displacement limits |
| Frozen policy | Same checkpoint and action interface before/after on reserved starts | Per-trial success, numerical configuration, trace/video, aggregate comparison |

The solver search sees only development physics diagnostics. It never optimizes
policy reward or looks at reserved evaluation trials. Failure of geometry, solver
or controlled insertion blocks later stages. A policy failure does not erase a
successfully qualified physics/control setup.

## Python

```python
import json
from pathlib import Path
from newton_calibration.qualification import Recipe, QualificationJob
from my_trusted_scene import create_backend

backend = create_backend()
recipe = Recipe.from_dict(json.loads(Path("recipe.json").read_text()))
job = QualificationJob.create(recipe, backend, "runs/precision-001")

job.status()                         # next_stage, next_action, separate outcome fields
job.advance(expected_stage="geometry")
job.advance(expected_stage="solver")
job.advance(expected_stage="controlled_insertion")
job.advance(expected_stage="frozen_policy")
# Or job.run(): advance in order until complete or blocked.

# Following interruption, with the exact same inputs/provider:
job = QualificationJob("runs/precision-001", backend)
job.run()  # reuses committed experiment receipts; incomplete attempts are preserved/retried
```

The backend factory is trusted executable code installed by the operator. Never
load a factory/module/command supplied by an evidence file, model response or recipe.

## CLI

```bash
newton-calibration qualify create --recipe recipe.json --job runs/precision-001 \
  --backend-factory my_trusted_scene:create_backend
newton-calibration qualify advance --job runs/precision-001 --expected-stage geometry \
  --backend-factory my_trusted_scene:create_backend
newton-calibration qualify run --job runs/precision-001 \
  --backend-factory my_trusted_scene:create_backend
newton-calibration qualify status --job runs/precision-001 \
  --backend-factory my_trusted_scene:create_backend
```

Blocked gates exit 2. Worker failures propagate a nonzero exit and retain logs.
The CLI prints finite JSON; it does not require a chat agent. `create` refuses an
existing job directory. Changing inputs, solver plan, criteria, runtime/provider
or controller requires a new job/revision and baseline.

## Agent and optimizer integration

Minjae's agent can orchestrate `status/advance/run`; a solver specialist can propose
the bounded recipe before it is locked. Neither is a dependency of the core.
**This feature does not install or impersonate Minjae's private agent.**

The framework-neutral `AgentTools` wrapper exposes three narrow host tools. Bind
the backend outside the model's arguments, then register these methods with the
agent framework used by your team:

```python
from newton_calibration.qualification import AgentTools

tools = AgentTools(backend=create_backend(), jobs_root="runs/qualification")
tools.describe()  # capabilities, tool argument descriptions and operating rules
tools.create(job_id="tight-fit-001", recipe=recipe_json)
tools.status(job_id="tight-fit-001")
tools.advance(job_id="tight-fit-001", expected_stage="geometry")
```

`advance` refuses stage skipping. Job IDs cannot traverse arbitrary directories.
The agent host owns durable scheduling and authorization; this wrapper is not
a new chat service. Solver agents propose data-only candidate settings before
locking a recipe; runtime fitting is still executed by the toolkit.

The default search is an explicit ordered list of numerical candidates. To use an
installed optimizer instead, set recipe `optimizer` and leave `candidates` empty:

```json
{
  "name": "minjae-nvopt.v1",
  "parameters": [{
    "name": "margin_m", "lower": 0.0, "upper": 0.00005,
    "initial": 0.000025, "unit": "m", "owner": "simulator",
    "rationale": "Bounded contact-surface-offset hypothesis; not material calibration"
  }],
  "options": {},
  "seed": 42
}
```

The existing `newton_calibration.optimizers` entry-point registry is reused.
A missing provider fails explicitly; the built-in optimizer is not silently
substituted. This preview requests **one candidate per ask/tell generation**;
the provider must support population 1. Continuous optimizer terms are limited
to `margin_m`, `gap_m` and `tolerance`. Timestep and iteration alternatives belong
in explicit sweeps; no silent rounding of optimizer proposals. Search stops at
the first qualified candidate or the declared budget, not a claimed global optimum.

Solver acceptance includes an independent repeat and a refinement with half the
physics timestep, doubled iterations (capped at 1000) and 10× tighter tolerance,
bounded by the backend's declared tolerance floor. The locked backend description
and each candidate record make that floor and actual refinement settings explicit.
At the floor, convergence tests refine timestep and iterations, not tolerance.
The policy command frequency remains fixed through integer decimation. This is
a declared convergence check, not proof of the continuous-time physical solution.

## Backend contract

`Backend.describe()` reports actual runtime/source identity, operation support,
parameter bounds, input dependency paths and content fingerprints for each start.
`Backend.execute(request, output_dir)` runs one experiment and returns measurements,
applied settings, input hashes, policy cadence and relative artifact paths.

The toolkit determines pass/fail, validates exact trial sets and finite values,
checks bounds, hashes input files before/after execution and rejects input drift.
Readback and operation implementations remain the responsibility of the trusted
backend; the protocol is not a sandbox for malicious plug-ins. `CommandBackend`
provides shell-free subprocess execution, per-experiment timeout and process-group
termination for its own worker only. Direct in-process providers must implement
their own bounded execution; Python cannot safely kill an arbitrary in-process call.

### Durable records

- `plan.json`: sealed recipe, gates, input hashes, runtime/provider identity and splits.
- `experiments/<id>/attempt-*/`: invocation, traces, requested videos, worker logs or failure.
- `experiments/<id>/receipt.json`: completed request/result and artifact hashes.
- `search/`: optimizer proposals and post-tell checkpoints, when used.
- `stages/`: immutable stage decisions bound to the plan.
- `status.json`, `REPORT.md`: user/agent summary and next action.
- `progress.json`: current/last experiment and attempt; `status()` reports a
  running worker when the job lock is held instead of suggesting a duplicate run.
- `qualified_simulation.json`: scoped numerical settings only after the first
  three stages pass, with the separate frozen-policy result. **Not** a real-data
  calibration overlay accepted by the existing calibration package loader.

The file lock prevents concurrent orchestrators from running the same job. Resume
is at completed-experiment boundaries, not mid-GPU-rollout; interrupted attempts
may run again and are retained. Hashes detect accidental modification/mixed jobs;
they are not signatures or a security trust anchor against a malicious writer.

## Current reference and limitations

The first adapter is `projects/flexiv_resin_insertion/qualification`: pinned
Isaac Lab 3.0.0-beta2 / Newton 1.2.1 / MuJoCo Warp, the supplied 25 mm cylindrical
resin peg and faceted straight/chamfered fixture, known target pose, physical grasp.
It reuses the existing frozen scene; another arm/asset needs its own trusted
adapter, geometry family checks, controller diagnostic and validation metrics.
The core itself has no Flexiv IDs, joint counts or scene paths.

The reference adapter applies contact margin/gap to the **peg and fixture only**
through the pre-finalization model-builder callback. It preserves robot/gripper/
table collision settings and records imported vs applied values in
`shape_binding.json`. Finalized model arrays and MuJoCo Warp options are read back.
The before condition is the explicit recipe baseline, not an automatic claim of
bit-identical contact settings to the original training run.

The pinned MuJoCo Warp runtime clamps solver tolerance to at least `1e-6` during
model conversion. This adapter advertises that minimum and rejects tighter recipe
values before execution; it does not patch the runtime or claim a tighter setting
was applied. Final active options must still match the declared settings.

The reference inspector accounts for faceting and checks all axial wall slabs.
Runtime validation uses a conservative enclosing-cylinder/inscribed-bore bound
along the peg at physics cadence. It can reject a physically valid mesh fit;
it is not exact mesh penetration, contact-force accuracy or deformable-resin
validation. Unsupported topology/transforms fail rather than invent precision.
Friction/density are still unmeasured assumptions. No real transfer is established.

Recorded videos show the first **preselected** trial in the declared split;
trace/metrics include every requested trial, including failures. Rendering uses
RTX only; physics stays Newton. Missing requested video blocks completion.

Local tests use explicitly labeled `test_double` measurements and are not Newton
physics evidence. Live run outcomes must be reported separately from test passes.
