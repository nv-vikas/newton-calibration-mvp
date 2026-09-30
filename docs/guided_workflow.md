# Pick a recipe. Follow one workflow.

The user should not have to assemble the five calls or switch to a separate
experiment when data is missing. The guide stays with one session from the first
question through collection, fitting, validation and the customer report.

## The experience

| User | Guide |
|---|---|
| “What recipes can I run?” | Arm joint tuning is available. Grasp and insertion are planned. |
| “Tune this robot for my task.” | Inspects the USD; proposes joint names and a parameter scope. |
| “Here is my controller configuration and gripper setup.” | Records the source; asks only for unresolved details and explicit confirmations. |
| “I have some recordings.” | Inspects signals, joint mapping and training coverage; reuses sufficient evidence. |
| “I do not have enough data.” | Prepares supported collection motions once the scene and simulation envelope are known. Requests the exact-command video by default. |
| “The engineer collected these logs.” | Attaches them to the same session, creates a new revision and rechecks readiness. |
| “Run calibration.” | Executes the existing guarded five calls. Saves a package and customer report, including failures and limits. |

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

# Inspect and prepare collection, or stop at a ready-to-fit plan.
newton-calibration guide advance --session runs/my-arm
# Only this explicit request allows fitting, validation and packaging.
newton-calibration guide advance --session runs/my-arm --execute
# Nonmutating snapshot; works while fitting holds the session's writer lock.
newton-calibration guide status --session runs/my-arm
```

`--confirmed-by` alone confirms nothing. The answers must explicitly list which
items were reviewed, e.g. `"confirm": ["mapping", "controller", "tool", "bounds"]`.
Record the actual verifier, not an agent-generated identity. These are attributed
assertions, not an authentication system or hardware authorization.

## After Analyze: what do I do next?

Analysis can complete while fitting is not ready. Every newly saved session has
an `action_plan` and a short `next_action`; the CLI displays priorities, owners
and **done-when** criteria instead of an undifferentiated warning list.
Detailed input questions remain in JSON and the interactive wizard.

For example, when existing joint recordings are usable for inspection but replay
timing and setup confirmation are unresolved:

| Priority | Owner | Action | Done when |
|---|---|---|---|
| 1 | Toolkit engineering / agent | Qualify the replay adapter | Original command timing and applicable limiter/filter behavior are tested at the declared physics step. |
| 2 | Agent | Prepare the configuration review sheet | Proposed simulation gains, limits, bounds and assumptions cite their sources; unknowns remain explicit. |
| 3 | Robot engineer, guided by agent | Review unresolved setup facts | Joint mapping, real controller and tool/payload declarations name a verifier and source. |
| 4 | Toolkit | Re-analyze existing recordings | All gates pass before a fitting plan is locked; otherwise the next specific gap is reported. |

**Data collection: deferred.** First address the replay/setup questions. Do not
ask the engineer to repeat collection just because software cannot replay it yet.
Do not alter the declared physics timestep merely to hide a rate mismatch.
The current adapter's multirate limitation remains; this action list does not fix it.

The data decision can be `not_assessed`, `deferred`, `needed`, or `not_requested`.
Missing/insufficient evidence leads to targeted collection, subject to setup and
operator review. Unreadable existing evidence leads to interpretation first.
Screening eligible parameters is not proof that their values are identifiable.

Agents read `action_plan.items`: stable ID, priority, owner, action, `done_when`,
`source_questions`, `depends_on`, and `status: proposed`. Actions are derived from
the current session revision; they are **advice, not a task executor or completion
record**. Only the existing readiness checks allow progress. Fitting still needs
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
session = guided.advance("runs/my-arm", preview=bound_preview,
                         design_probe=bound_probe)  # no fit yet
session = guided.advance("runs/my-arm", execute=True)
snapshot = guided.status("runs/my-arm")
```

Every CLI operation also supports `--json` except the interactive wizard. The
structured result has `questions`, `proposals`, `scope`, `evidence_needs`,
`fit_readiness`, `action_plan`, artifact references and an explicit next state.

### Answer sections

| Section | Contents |
|---|---|
| `environment` | `EnvironmentSpec`: source-to-USD map, groups, per-joint PD/effort baselines, bounds, dt and fixed runtime settings. Confirmation booleans here are ignored. |
| `controller` | Real/sim modes, command semantics, units, rate, filtering, compensation and source revision. |
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
instead of substituted with PD motions. The current fit adapter requires matching
command and physics-step rates; multirate replay is not silently assumed.

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
