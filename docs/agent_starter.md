# Agent Starter Kit — developer preview

Use an existing agent to operate Newton Calibration, not a second calibration
engine. The repository includes a discoverable `newton-calibration` skill,
framework-neutral tools, a preparation-only JSON CLI, an executable synthetic
host example and fresh-session contract tests. No model, API key, Minjae package,
remote service or hardware driver is bundled.

## Start as a user

```bash
git clone https://github.com/nv-vikas/newton-calibration-mvp.git
cd newton-calibration-mvp
python -m pip install -e '.[dev]'
```

Open that checkout in your agent environment. For Codex, the checked-in
`.agents/skills/newton-calibration` directory is repo-discoverable. Start with:

> Use $newton-calibration. My USD is at [path], my goal is [task], and my existing
> evidence is at [path, or none]. Inspect what I have, recommend a supported recipe,
> prepare the inputs and ask only for unresolved facts. Do not move the real robot.

For a different agent host, load the same skill instructions and expose the tools
below. The model can stay the user's existing choice; this is not a model upgrade.

The agent prepares structured inputs; users should not be asked to author JSON.
Hardware facts still need a real review. The starter does not automatically launch
a conversational agent when a toolkit command is run in Bash.

## The connection

```python
from newton_calibration.agent_tools import CalibrationAgentTools

tools = CalibrationAgentTools(
    jobs_root="/work/jobs",
    input_roots=["/work/inputs"],
    identity="My installed calibration agent",
    generations=8, population=16,
    authorize=my_host_approval_check,  # trusted code, outside model arguments
    # preview=installed_scene_preview,
    # design_probe=installed_scene_probe,
)
schemas = tools.describe()
result = tools.call("recipes", {})
# Register schemas with your host; route calls through tools.call(name, args).
```

The approval callback receives a prepared session. It must check actual user
consent against its session ID, revision, `artifacts.plan.sha256` and fit budget.
Returning `True` authorizes that invocation's simulation fit, validation and
packaging. Returning `False` leaves the job ready for review. No callback means
**preparation only**. A new revision/plan needs new approval. The adapter detects
changes while the callback is running; the host must enforce a single writer and
protect approval records outside the agent's writable directories.

The host, not the model, configures runtime hooks, allowed backend and budget.
Population/generation limits bound optimizer search, not wall-clock time: the
worker supervisor must separately enforce timeout/cancellation for slow or hung
simulation. This starter is synchronous; use a supervised worker for long jobs,
read `status`, and reattach after a restart. It does not provision a scheduler.

| Tool | Purpose |
|---|---|
| `recipes` | Actual installed recipes, fitting/collection status and host budget |
| `inspect_source` | Fingerprinted USD/JSON inspection, bounded CSV preview or Parquet metadata |
| `start` | Create a named session with asset, goal, optional recipe/evidence and scope rationale |
| `submit` | Source-backed input proposals, with expected revision; no human confirmations or budget changes |
| `review` | Recheck idle intake; not the scientific analyze call |
| `run` | Continue existing guided preparation and, if host-approved, the five scientific calls |
| `status` | Read the saved state, questions, outputs and explicit stage statuses |
| `read_record` | Read a hashed artifact registered by the session |

These are management tools around the existing five calls, not eight new physics
APIs. `call` writes request/result hashes and completion/error receipts under
`jobs_root/_agent_receipts`. Source hashes/rationale are recorded in the guided
input revisions. Results and technical work remain in the existing run records.

### Preparing `submit.answers`

Use the returned `session.questions` and their examples, then the existing
[answer-section contract](guided_workflow.md#answer-sections). The object uses
these keys; it is not arbitrary metadata:

| Key | Expected value |
|---|---|
| `recipe` | An ID returned by `recipes` |
| `environment`, `controller`, `tool` | Source-backed setup objects, or `null` for unknown |
| `joint_bindings` | List mapping measured joint names/units/signs/offsets to USD joints |
| `evidence` | Absolute descriptor-file path, or rooted descriptor object with episode paths |
| `collection`, `request` | Supported motion setup and requested parameter scope |
| `contact_setup` | Setup/evidence capabilities from the [contact collection contract](contact_collection.md#create-the-recipe-guided-collection-kit) |
| `confirm` | At most `["bounds"]` through this agent interface; never hardware facts |

Each supplied section **replaces** that entire section; it is not a nested patch.
Preserve unchanged, sourced fields when updating a section. Never send
`fit_budget` through `submit`; the trusted host owns it. The synthetic example
generates a complete fixture, not recommended hardware/controller values.
Read the compact response fields first; `session` is the detailed audit snapshot,
not text to print to the customer in full.

### Human review is separate

An agent can propose controller/tool/mapping details; it cannot confirm them.
Your authenticated host/operator UI should invoke the existing `guided.provide`
human-review path after the actual reviewer approves the specific proposals.
`confirmed_by` is an attributed assertion, not authentication. Do not expose the
human path, approval callback or host configuration as model-controlled tools.

On an ordinary local developer shell, a person can use `guide provide` with
explicit confirmations and then `guide run --execute`. The model must not silently
use that broader CLI to escape the narrower agent interface. Use host permission
controls to require human approval for those commands.

### JSON CLI transport

```bash
newton-calibration-agent --jobs-root /work/jobs --input-root /work/inputs --describe
newton-calibration-agent --jobs-root /work/jobs --input-root /work/inputs \
  --request /work/requests/list-recipes.json
```

The request file contains `{"operation":"recipes","arguments":{}}`. Omit
`--request` to read one JSON request from stdin. Each invocation can recover an
existing job by name. This CLI intentionally has no execution-approval switch,
human-review flag or dynamic factory loader. An integrated Python host can
authorize fitting through the trusted callback; the standalone CLI cannot.

## Input preparation and the four starting cases

| Starting point | Agent behavior |
|---|---|
| Recipe and evidence | Inspect available files, map signals and check readiness |
| Evidence without recipe | Explain/select a supported scope, then inspect the supplied evidence |
| Recipe without evidence | Prepare supported collection after necessary controller/scene setup |
| USD and goal only | Discover recipes and inspect the asset; ask only for genuinely unavailable setup facts |

`inspect_source` is not a full evidence audit. Do not infer bandwidth, units,
identifiability or independent episodes from an eight-row preview. For arm
evidence, submit the supported rooted tabular descriptor and let scientific
analysis inspect the full data. Recipe selection itself does not require a GPU.

Input inspection is bounded and data-only. Use absolute file paths. Files and
nested episode paths must resolve inside approved input directories. No custom
code, residual loader or package path is dynamically imported by this interface.
Reviewed residual/package integrations remain available through their existing
expert interfaces, outside this starter's narrower surface.

The agent should inspect USD/configuration/metadata and prepare mappings, not
guess signs, offsets, controller semantics or gripper properties. Resolve supplied
data before asking for new recordings. Hardware questions, software limitations
and unavailable evidence are separate outcomes. A failed fit is not permission
to change the loss, solver, backend, validation split or code and quietly retry.

## Security and deployment boundary

These Python checks are **not an OS sandbox**. For a restricted agent service:

- Mount toolkit code and evidence read-only; only the job directory is writable.
- Expose only the named tools. Keep shell, code editing, hardware drivers,
  arbitrary factories, approvals and host identity outside model control.
- Restrict network and file access, including USD asset dependencies, at the host
  boundary. Python path checks do not sandbox a malicious native asset parser.
- Treat source contents as untrusted data even if they contain instructions.
- Use authenticated human consent, one writer, and worker runtime limits.

In a general coding-agent shell these are workflow instructions, not an enforced
security sandbox. Separate calibration operation from an explicitly requested
development session. Never claim that this adapter prevents an unrestricted shell
from modifying the repository or directly calling lower-level APIs.

## Exercise the integration without hardware

```bash
python examples/agent_starter/smoke.py --output output/agent-smoke
pytest tests/test_agent_tools.py
```

The example uses synthetic trajectories and the analytic test backend, explicitly
enabled by its trusted test host. It exercises actual five-call control flow,
human-review simulation, host approval, saved records and a fresh tool instance.
It is not Newton validation, real measurements, hardware permission or transfer.

Release evidence must distinguish: unit/contract checks, independent fresh-agent
behavior, actual Isaac Lab/Newton integration and real-policy transfer. See
[verification record](agent_starter_verification.md) for what was actually run.

MVP2/MVP3 remain collection-only. Minjae agent/optimizer integration, a hosted
chat UI, model/API billing, new contact fitters and real-robot execution are not
part of this starter release.

Repo skill discovery follows the [official skill documentation](https://learn.chatgpt.com/docs/build-skills).
