# Pick a recipe. Follow one workflow.

The user should not have to assemble the five calls or switch to a separate
experiment when data is missing. The guide stays with one session from the first
question through collection, fitting, validation and the customer report.

## The experience

| User | Guide |
|---|---|
| “What recipes can I run?” | Arm joint fitting is available. Grasp and insertion prepare collection specifications only. |
| “Tune this robot for my task.” | Inspects the USD; proposes joint names and a parameter scope. |
| “Here is my controller configuration and gripper setup.” | Records the source; asks only for unresolved details and explicit confirmations. |
| “I have some recordings.” | Inspects signals, joint mapping and training coverage; reuses sufficient evidence. |
| “I do not have enough data.” | Prepares supported collection motions once the scene and simulation envelope are known. Requests the exact-command video by default. |
| “The engineer collected these logs.” | Attaches them to the same session, creates a new revision and rechecks readiness. |
| “Run calibration.” | Executes the existing guarded five calls. Saves a package and customer report, including failures and limits. |

The scientific-call rows above describe the supported **arm recipe**. For
`grasp_contact@1` and `peg_insertion@1`, the same `start → provide → run`
interface prepares a [contact collection bundle](contact_collection.md) and
stops at `collection_spec_prepared`. It does not call scientific analyze/plan,
fit, validate or write; all five are explicitly `not_run`. Even `--execute`
does not enable an unavailable contact fitter or robot execution. Existing
contact evidence is retained but is not yet parsed/audited by a contact adapter.

All four starting cases converge here: recipe/data supplied or missing, with an
asset and goal. “Unknown” is a valid intake answer, never a confirmation.
Unreadable existing data is not silently treated as missing data.

## Start from the terminal

```bash
python -m pip install -e '.[dev]'
newton-calibration guide
```

The terminal wizard is deliberately thin. It asks simple questions and accepts
JSON or a JSON-file path for detailed setup answers. An external agent can
translate the engineer's configuration into those answers, show its proposals,
and ask the same questions conversationally. There is no bundled LLM or claim
that Minjae's agent is already integrated.

For an explicit, resumable start:

```bash
newton-calibration guide recipes
newton-calibration guide start \
  --asset /path/to/robot.usd \
  --goal "Prepare the arm for the insertion task" \
  --recipe arm_joint_response@1 \
  --session runs/my-arm
newton-calibration guide review --session runs/my-arm

# Data-only answers; no hardware control or Python execution from this file.
newton-calibration guide provide --session runs/my-arm \
  --answers /path/to/reviewed-answers.json \
  --source "Controller config revision and engineer review" \
  --confirmed-by "Engineer who verified the listed confirmations"

# Automatically prepare what is supported, then stop at a ready-to-fit plan.
newton-calibration guide run --session runs/my-arm
# One authorization covers fitting, validation and packaging; no per-step prompts.
newton-calibration guide run --session runs/my-arm --execute
# Nonmutating snapshot; works while fitting holds the session's writer lock.
newton-calibration guide status --session runs/my-arm
```

`--confirmed-by` alone confirms nothing. The answers must explicitly list which
items were reviewed, e.g. `"confirm": ["mapping", "controller", "tool", "bounds"]`.
Record the actual verifier, not an agent-generated identity. These are attributed
assertions, not an authentication system or hardware authorization.

An agent may choose and record **simulation search bounds** under its own identity,
using `provide(..., reviewer_kind="agent", confirmed_by="actual agent name")`
or CLI `--reviewer-kind agent`. Only `"confirm": ["bounds"]` is permitted in
that case. It cannot confirm mapping, real controller or attached-tool facts.
This does not override any readiness check or authorize real robot execution.

### Work the toolkit/agent can finish without asking the operator

`guided.run` executes built-in preparation rather than returning it as user
homework. It fills **missing** simulation baseline maps from an explicitly supplied
`--simulation-profile /path/to/profile.json`, selects the installed explicit-PD
simulation mode when declared real command semantics are compatible, and records
an agent review of recipe/declared simulation search bounds. It then calls the
existing guarded workflow. Existing baseline declarations are not overwritten;
raw USD angular gains are not copied into a radian-based controller. A changed
setup conservatively invalidates previous setup confirmations.

An explicitly trusted host agent can extend preparation:

```python
from newton_calibration import guided

result = guided.run(
    "runs/my-arm", execute=True,
    simulation_profile="/path/to/known-simulation-profile.json",
    resolver=my_installed_agent_resolver,  # optional trusted callable
)
```

The resolver receives a copy of the session and returns `{"answers": {...},
"source": "configuration path / revision / evidence"}` or `None`. Answers use
the same validated `provide` contract. A resolver cannot change the selected
recipe, declared backend, task scope or fit budget, or confirm hardware facts.
Recipe and evidence files cannot load Python code. The resolver is not an LLM
bundled with the toolkit and does not independently authenticate its sources.

Preparation is bounded (six passes by default), detects no progress and checks
the revision before applying answers. Every applied change records provenance
and preserves the previous revision. The resulting `continuation` record says
`completed`, `awaiting_user`, or `toolkit_attention`. The last means an unresolved
source/software limitation—not a running background job. Invalid source files or
resolver responses raise errors rather than triggering a fallback fit.
`advance` remains available as the lower-level deterministic workflow API.

`guided.preflight` offers offline helpers for source-backed preparation:

- `inspect_simulation_profile(path, joint_map)` checks explicit per-joint
  baseline values from an existing toolkit profile and records its hash. It
  produces a proposal, not hardware settings or a runtime validation.
- `inspect_usd_properties(path)` inventories authored joint drives and body
  mass/inertia without running Newton. Missing properties stay missing. Raw
  angular drive gains are not copied into a radian-based controller.
- `replay_slew_limiter(...)` models a declared velocity limiter using command
  event times. It is diagnostic only. Initial limiter state and actual send
  times are required for an exact historical replay; a reconstruction must
  never be relabeled as measured published commands or used to bypass intake.

Record completed checks, changed simulation choices, source hashes and remaining
questions with the new run. Preserve original recordings and prior sessions.
Ask the operator only for unresolved setup facts or existing missing files—not
for simulation gains that already have a source or a blanket new data collection.

Replay sample timing is versioned: sample zero is the initial state at `t0`;
command `k` advances the simulator to sample `k+1`. N recorded samples require
N−1 transitions, not N. The fit execution fingerprint includes this convention,
so a pre-fix fit journal cannot resume silently with changed scoring semantics.
Historical results are retained; they are not upgraded by these software tests.

## After Analyze: what do I do next?

Analysis can complete while fitting is not ready. `guide run --execute` continues
supported work until completion or a real boundary. The customer CLI/report shows
`action_plan.user_requests`: only unresolved setup facts, recipe/execution choices
or the operator's real collection. It does not ask the user to implement adapters,
set simulation defaults or run each scientific call separately.

For example, when existing joint recordings are usable for inspection but the
driver's command limiting and setup confirmation are unresolved:

The following remains an **internal** work queue, behind technical details:

| Priority | Owner | Action | Done when |
|---|---|---|---|
| 1 | Toolkit engineering / agent | Resolve requested versus driver-published targets | Actual published targets are bound, or the limiter/filter replay is separately implemented and qualified. |
| 2 | Agent | Prepare the configuration review sheet | Proposed simulation gains, limits, bounds and assumptions cite their sources; unknowns remain explicit. |
| 3 | Robot engineer, guided by agent | Review unresolved setup facts | Joint mapping, real controller and tool/payload declarations name a verifier and source. |
| 4 | Toolkit | Re-analyze existing recordings | All gates pass before a fitting plan is locked; otherwise the next specific gap is reported. |

**Data collection: deferred.** First address the replay/setup questions. Do not
ask the engineer to repeat collection just because software cannot replay it yet.
Do not alter the declared physics timestep merely to hide a rate mismatch.
Different command and physics rates are supported by timestamped target holding;
that does not automatically reconstruct missing filtered/limited targets.

The data decision can be `not_assessed`, `deferred`, `needed`, or `not_requested`.
Missing/insufficient evidence leads to targeted collection, subject to setup and
operator review. Unreadable existing evidence leads to interpretation first.
Screening eligible parameters is not proof that their values are identifiable.

Agents can inspect `action_plan.items`: stable ID, priority, owner, action, `done_when`,
`source_questions`, `depends_on`, and `status: proposed`. Actions are derived from
the current session revision; they are **advice, not a task executor or completion
record**. `guided.run` records preparation it actually performs separately in
`continuation.automatic_preparation`; the five-call artifacts remain the results.
Only the existing readiness checks allow progress. Fitting still needs
an explicit execution request; hardware motion is never authorized by this list.
Older saved sessions acquire the new list on their next review/advance. Read-only
`status` does not rewrite them. A recipe digest change still requires a reviewed
new session revision rather than silently reusing an old fitting plan.

## What happens when evidence is missing?

1. **Analyze completes** with evidence needs; fitting remains unavailable.
2. The guide asks for the selected scene, supported controller, starting pose,
   controlled joints, simulation limits, amplitudes and velocity/acceleration caps.
   It must not infer safe real-robot limits from a USD.
3. `plan(intent="collect")` selects supported experiments for the missing
   parameters. It writes timed command CSVs, hashes, expected signals, instructions,
   deferred requirements and a default video request. If only holdouts are missing,
   it requests holdouts instead of repeating the training campaign.
4. An installed preview/probe adapter runs those exact commands in Isaac Lab with
   Newton. No adapter means **preview pending**, not “screened” or a fake video.
5. The operator reviews the proposals, real workspace/controller limits and stop
   procedure, and performs approved hardware collection outside this guide.
6. Attach the new evidence using `provide`. Re-analysis decides what can now be
   fitted. Generated motions alone never prove that every parameter is identifiable.

The existing parameter-aware catalog provides settling, chirps/sweeps, reversals,
acceleration-rich motions and distinct holdouts as appropriate. Missing clock
semantics, torque instrumentation or saturation evidence are not repaired by
blindly generating more motion. Requested parameters are never silently dropped.

### Connect a running Isaac Lab scene

The application still owns scene creation, Newton initialization and cameras.
Use `IsaacLabScenePreview` from `newton_calibration.collection.isaaclab_preview`
and `FiniteDifferenceProbe(IsaacLabPredictionBackend(...))` from the collection
sensitivity/Isaac Lab probe modules. Supply explicitly trusted, installed factories:

```bash
newton-calibration guide advance --session runs/my-arm \
  --preview-factory my_scene:make_preview \
  --design-probe-factory my_scene:make_probe
```

Each factory returns a bound callable; both must use the same initialized scene
and reviewed motion profile. Factories are CLI/application code, never executed
from recipe, evidence or answer JSON. The preview adapter checks the source USD,
scene identity, Newton backend, command fingerprint and output artifacts. It uses
a pinned Isaac Lab integration, not a promise of arbitrary-version support.
Simulation screening is not hardware safety certification. No real robot driver
is connected by the guide.

## Agent / developer interface

These management functions wrap, rather than replace, the five scientific calls:

```python
from newton_calibration import guided

session = guided.start(asset="/path/to/robot.usd", goal="Insertion preparation",
                       directory="runs/my-arm", recipe="arm_joint_response@1")
# Agent reads questions/proposals. User or engineer supplies/approves facts.
session = guided.provide("runs/my-arm", answers, source="Reviewed configuration",
                         confirmed_by="Actual reviewer")
session = guided.run("runs/my-arm", execute=True, preview=bound_preview,
                     design_probe=bound_probe)  # automatically continues when ready
snapshot = guided.status("runs/my-arm")
```

Every CLI operation also supports `--json` except the interactive wizard. The
structured result has `questions`, `proposals`, `scope`, `evidence_needs`,
`fit_readiness`, `action_plan`, artifact references and an explicit next state.

### Answer sections

| Section | Contents |
|---|---|
| `environment` | `EnvironmentSpec`: source-to-USD map, groups, per-joint PD/effort baselines, bounds, dt and fixed runtime settings. Confirmation booleans here are ignored. |
| `controller` | Real/sim modes, command semantics, units, rate, filtering, compensation and source revision. `command_stage` distinguishes `requested` from `published` targets when processing exists. |
| `tool` | Explicit no-tool declaration, or gripper/tool identity, held payload (or none), dynamics/mounting reference, USD match and source. |
| `joint_bindings` | Source and USD joints, units, sign, scale and offset. Proposals require confirmation. |
| `evidence` | Bound evidence JSON path/object, or an intake descriptor with `root`, `episodes`, long-form `schema` and `signal_bindings`. Partial training-only evidence is inspectable, not replayable until qualified. |
| `collection` | Existing `MotionSpec`: scene ID, joints, center, limits, amplitude/rate caps and source. Simulation proposals only. |
| `request` | Existing `CalibrationRequest`: exact parameter scope and collection design budget. |
| `fit_budget` | `generations` and `population`, passed to the existing fitting API. |

Providing a section replaces that entire section; it is not a recursive patch.
`null` means unknown. Examples of complete answers are generated by the synthetic
demo below. Production values must come from the actual deployment configuration.

The default scope is stiffness/damping for each declared joint group. Explicitly
request other supported parameters with reviewed bounds and evidence. This is
effective response calibration, not guaranteed recovery of physical constants.
Current guided replay supports joint-position PD and absolute joint commands,
including a declared real joint-impedance path; this does not prove equivalence
to a vendor's internal controller. Cartesian/OSC/IK/torque commands are rejected
instead of substituted with PD motions.

### Slow commands, fast physics

The command rate does **not** have to equal the physics rate. For example, a
30 Hz target stream can run through 960 Hz physics: each target is held for about
32 ticks. The implementation uses actual timestamps, not a fixed repeat count,
so non-integer ratios and irregular arrivals are supported too. A transition is
applied on the first physics tick at or after its timestamp; quantization is less
than one physics step. Command arrivals closer together than one physics step
are rejected instead of silently discarded. Review a finer step and establish
a new baseline for such a case; do not slow Newton down to equal a slower stream.

The source recordings and declared `dt` remain unchanged. Analyze records a
`command_replay_timing_supported` gate and per-stream timing diagnostics; the
guided intake saves the policy, rates and timing check in `command_replay`.
Feedback interpolation does not create additional real measurements. These
checks establish command scheduling support, not real-controller equivalence,
clock synchronization, or physical calibration accuracy.

When `filters` is not explicitly `none`, requested targets are not accepted as
published targets merely because timing works. Declare `command_stage=published`
only when the bound recordings actually contain final driver-published targets
and the source has been reviewed. Otherwise the `controller_processing` question
remains: inspect existing logs or separately implement and qualify the processing
path. The current adapter does not reconstruct Flexiv's slew limiter. Do not
relabel or silently modify requested commands to clear this check.

An older saved guided session must reselect the revised recipe and re-analyze;
its original records stay intact. No earlier calibration result becomes qualified
because this scheduling check passed.

## State and traceability

| State | Meaning |
|---|---|
| `choose_recipe` / `needs_information` | Input or confirmation needed; no fitting. |
| `needs_evidence_description` | Existing recordings need interpretation, not replacement. |
| `needs_collection_setup` | Data is needed; scene/controller/envelope incomplete. |
| `collection_needs_review` | Collection prepared, but preview/design incomplete or failed. |
| `awaiting_operator_review_and_real_data` | Simulation artifacts available; human review and real data still required. |
| `ready_to_fit` / `ready_to_resume` | Readiness passed; explicit execution request required. |
| `needs_attention` | Failure recorded; no fallback. Resolve and resume, or revise changed inputs. |
| `completed` | Run/report written. Validation and activation decisions are separate and may be false. |

One session has one writer. Atomic `session.json` snapshots are readable while
running. Input changes create a new revision, retain old files and remove old
results from the current view. Confirmations are bound to setup fingerprints;
file or artifact changes fail closed. The fit API retains its existing candidate
journal/checkpoints. There is no automatic cloud scheduler or real-robot executor.

`guided_intake.json` records the recipe snapshot, task goal, setup declarations,
confirmations and session revision. Scientific results remain in `analysis.json`,
`plan.json`, `fit.json`, `validation.json` and the package manifest. The report
reads those results and shows missing/failed checks without inventing success.
Historical exploratory fits are not imported as completed guided calibrations.

## Try and test without a robot

```bash
python examples/guided/demo.py --output output/guided-demo
pytest tests/test_guided_workflow.py tests/test_reporting.py
```

The demo creates synthetic logs, prepares a collection proposal without a GPU
preview, attaches those logs to the same session, and runs all five calls on the
analytic test backend. It creates a customer HTML report and a non-activatable
test package. **This verifies software flow, not Newton physics, robot calibration
quality or real-task transfer.** Never send the fixture motions to hardware.
