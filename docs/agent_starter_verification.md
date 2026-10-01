# Agent Starter Kit — verification record

Date: 2026-09-30. Release status: **developer preview**.

This update packages the existing calibration workflow for an installed agent.
It does not add a new physics engine, new real-robot results, a Minjae integration
or grasp/insertion fitting. The repository is now
[nv-vikas/newton-calibration-mvp](https://github.com/nv-vikas/newton-calibration-mvp).
The Python package and existing command names remain compatible.

## What was checked

| Check | Result | What it establishes |
|---|---|---|
| Full Python test suite | 470 passed | Existing behavior and new agent-tool contracts |
| New agent-tool test module | 28 cases included above | Four starting cases, sourced proposals, approval boundaries, stale-plan rejection, failure/resume, path checks and capability limits |
| Five-call synthetic example | Completed; all five calls recorded | Analyze → plan → fit → validate → write can run through the wrapper after simulated human review and trusted host approval |
| Fresh tool instance | Same output artifact references; no repeated fit | Saved state can be reattached without relying on prior chat history |
| Independent fresh-agent walkthrough | **Partial**: catalog, source inspection and arm-session creation completed | Initial instructions can be followed without prior chat; end-to-end agent behavior is not yet qualified |
| Skill format validation | Passed | Repository skill metadata/layout is valid |
| Ruff checks | Passed | Changed Python modules meet the checked lint rules |
| Wheel installation / entry points | Passed with existing dependencies | Built wheel imports and both agent/toolkit command entry points work |
| Isaac Lab / Newton on Horde | **Not run: access blocked** | No new GPU/runtime qualification claim |
| Real robot / task transfer | **Not run** | No hardware execution or transfer improvement claim |

The synthetic example explicitly used the analytic test backend with generated
trajectories, two generations and population four. Its output identified
`newton_executed=false`, `real_data_used=false`, `activation_allowed=false` and
`resume_preserved_artifacts=true`. This is software-flow evidence, not a physics
accuracy result. The example is reproducible with
`python examples/agent_starter/smoke.py --output output/agent-smoke` using a fresh
output directory. Local generated fixtures/reports are not committed.

The wheel test used an isolated temporary virtual environment with existing
system dependencies. It did **not** establish clean dependency installation,
GPU/container compatibility or compatibility with every agent host.

## Independent agent walkthrough

A separate agent received the checked-in skill, the synthetic input directory
and two user goals: prepare arm calibration without a preselected recipe, and
prepare future peg-grasp collection with a wrist sensor/phone but no confirmed
jaw-force reference. The test host allowed only the analytic fixture backend,
two generations/population four, and no fitting-approval callback.

The saved transcript shows tool discovery, recipe discovery, four source
inspections (USD, setup JSON, training CSV and held-out CSV), and creation of an
arm session using `arm_joint_response@1`. The agent identified the inputs as
synthetic and submitted no human confirmations. All five scientific calls were
still `not_run` at the recorded checkpoint.

The walkthrough stopped making observable progress while its test harness was
preparing a submission request and was interrupted. No `submit` invocation was
recorded; this does not establish a toolkit hang. The grasp case was not reached.
This is **not a passed end-to-end agent test**
or a reliability measurement. The boundary/capability cases above are verified
by deterministic tests, not by claiming this independent walkthrough completed.
The reviewer also noted lengthy responses and sparse `answers` schema guidance.
The guide now links the existing answer contracts, lists accepted sections and
explains replacement semantics; a compact response redesign remains future work.
Repeat the complete agent-host walkthrough before a qualified agent release.

## Runtime qualification still required

The existing Horde endpoint redirected to a new service. The installed client
reported an expired login, and its normal login flow failed on the redirect.
No new node was provisioned and no Newton replay was executed for this release.

Before calling this runtime-qualified, restore access and run one bounded,
approved Isaac Lab/Newton calibration through this same wrapper with real
evidence. Verify the runtime fingerprint, held-out records, interrupted-run
recovery and packaged overlay. Keep that result separate from real-policy
transfer, which requires its own robot trials.

## Operational limits

- The host must supply authenticated human review, execution consent, backend
  policy, read-only input/code mounts and worker timeouts. Python checks are not
  an operating-system sandbox.
- The JSON CLI is preparation-only; a trusted integrated Python host can approve
  simulation fitting. Running Bash alone does not start a conversational agent.
- Input previews are not full evidence analysis. Unsupported controllers or
  absent runtime capabilities must remain explicit blockers.
- Grasp/insertion recipes currently prepare collection specifications only.
  Wrist force/torque is not a measurement of opposing-jaw squeeze force.

See the [starter guide](agent_starter.md) for setup and the exact tool boundary.
