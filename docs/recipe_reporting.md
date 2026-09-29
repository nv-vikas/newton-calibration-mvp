# Recipe-driven reporting

**Recipe = the questions. Run records = the answers. Report = the explanation.**

The main report is for the customer. It should answer: what improved, what work
was done, what remains unproven, and what to do next. Configuration, provenance
and detailed checks sit behind stage disclosures or in the downloadable records.

## Boundaries

| Artifact | Owns | Must not do |
|---|---|---|
| [Arm recipe](../src/newton_calibration/recipes/arm_joint_response.v1.json) | Reusable inputs, parameter/evidence relationships, collection families, required report facts | Contain a customer's fitted values or pretend every parameter is identifiable |
| Setup profile | Actual USD, controller path, tool/payload, mapping, limits and confirmation sources | Treat an agent's suggestion as operator confirmation |
| Locked execution plan | Exact supported parameters/bounds, objective, optimizer, runtime and split | Change mid-fit without a new baseline/revision |
| Run records | Executed work, measurements, checks, results, failures and provenance | Substitute plans or UI progress for completed experiments |
| Report bundle | Versioned links from semantic facts to hash-checked source records | Execute code, silently change records, or authorize activation |
| Customer HTML | Outcome, evidence, limits and next action | Imply transfer because the simulation error decreased |

The reporting contract is **currently a post-run attachment**. Its digest is
locked to the report bundle, not yet added to the calibration plan fingerprint.
The JSON recipe's procedural sections are documentation; reporting field
requirements are enforced by code. Existing Python recipes own executable
calibration behavior. This change does not implement new collection generators,
new controller support, grasp, or insertion calibration.

## Run the existing toolkit, then report

The scientific lifecycle stays **analyze → plan → fit → validate → write**.
Reporting is a read-only consumer, not a sixth scientific call.

```python
from newton_calibration.reporting import build_report, render_report
from newton_calibration.reporting.adapters import export_toolkit_run

recipe = "src/newton_calibration/recipes/arm_joint_response.v1.json"
bundle = export_toolkit_run(
    run_dir="runs/my-run",
    recipe_path=recipe,
    destination="output/my-report-revision-1",  # new directory
    package_dir="packages/my-package",         # omit when no package was written
)
model = build_report(recipe, bundle)
render_report(model, "output/my-report-revision-1/report.html")
```

This consumes `analysis.json`, `plan.json`, `fit.json`, `validation.json` and the
optional package `manifest.json`. It checks same-run identity and nested
plan/fit consistency. It never reads `status.json` as scientific evidence.
Partial runs render with missing facts and unknown stages. Missing records do
not prove a stage was never attempted: `unknown` and `not_run` are distinct.

The adapter selects held-out `position_rmse_rad` for the customer comparison.
The optimizer's weighted loss and the validation gate results are shown
separately. A position-only improvement does not override a failed weighted gate.
The integration test executes the real five API functions with the analytic
test backend and checks their serialized results; it is not a Newton GPU test.

## What a reporting requirement looks like

Inside the recipe:

```json
{
  "id": "baseline_error",
  "label": "Held-out baseline error",
  "stage": "validate",
  "type": "number",
  "required": true
}
```

Inside the run bundle:

```json
{
  "facts": {
    "baseline_error": {
      "record": "validation",
      "pointer": "/baseline_metrics/position_rmse_rad"
    }
  },
  "records": {
    "validation": {
      "path": "records/validation.json",
      "sha256": "<actual SHA-256 produced by the exporter>",
      "run_id": "my-run"
    }
  }
}
```

Pointers use JSON Pointer syntax: `/` traverses a key, `~1` escapes a literal
slash and `~0` escapes a tilde. No expressions, Python, shell commands or remote
URLs are evaluated. Facts must reference records; inline result literals are
rejected. An adapter may emit a **derived view record** for short explanations,
but it must retain the input records and record their original digests. Such
explanations are adapter interpretations, not new measurements.

The reporting contract asks for 33 facts, including:

- Readiness, controller/tool context, actual calibrated scope and split.
- Parameters, bounds/ownership, objective, optimizer and fixed runtime.
- Which motions were reused or generated, and preview availability.
- Held-out baseline/tuned metrics, gates and uncertainty/identification limits.
- Per-stage status, output scope, activation decision and next action.

No confidence estimate? Record that it was not estimated. No preview? Say so.
Do not insert an empty success story to satisfy completeness.

## Connect a new producer

1. Select/version a recipe and its reporting contract. Preserve the required core facts.
2. Snapshot the actual source records; keep original digests and run linkage.
3. Map semantic fact IDs to source record pointers. Add a named, versioned
   adapter view only for derived summaries or reorganized data.
4. Call `write_bundle(...)`, `build_report(...)` and `render_report(...)`.
5. Test missing evidence, failed validation, changed files, unknown states and
   a non-activatable result—not only the happy path.

```python
from newton_calibration.reporting import write_bundle

bundle = write_bundle(
    recipe_path=recipe,
    destination="output/new-snapshot",
    run_id="my-run",
    execution_kind="public_api",  # or legacy_import / synthetic
    records={"validation": recorded_validation},
    facts={
        "baseline_error": ("validation", "/baseline_metrics/position_rmse_rad"),
        "tuned_error": ("validation", "/calibrated_metrics/position_rmse_rad"),
    },
)
# This intentionally produces an incomplete report until other required facts
# are bound. Nothing is filled from expected recipe outcomes.
```

`--strict` on the renderer CLI exits with code 2 for missing required facts,
**after writing a report that shows the gaps**. It does not indicate scientific
failure or success; the recorded validation decision does that. Malformed data,
changed hashes or a mismatched recipe digest fail before rendering.

## Import the historical Flexiv experiment locally

```bash
python examples/reporting/import_flexiv.py \
  --original /path/to/original-fit \
  --replay /path/to/frozen-replay \
  --audit /path/to/audit_report.json \
  --media /path/to/local/trace-gifs \
  --output output/flexiv-report-revision-1
```

`--media` is optional and expects `visual-joint-1.gif` and `visual-joint-4.gif`.
The importer checks that the replay references the original plan, result and
unchanged parameters. It uses the declared primary replay variant, not whichever
variant looks best. The original search and later zero-optimizer replay remain
separate. Its main report says **experimental import**, not five successful API
calls; numerical success never becomes a qualified package or real-task proof.

The local customer report may contain source paths and other sensitive details
inside its downloadable records. **Review the entire embedded archive before
sharing externally.** This repository publishes only synthetic UI examples,
not the user's Flexiv records, videos, hardware photos or generated report.

## Trust, media and versioning

- Digests detect mutation relative to a bundle. They are not signatures and do
  not establish producer identity or accuracy of a supplied claim.
- Reports cannot activate anything. Use the existing verified package loader,
  with an independently supplied manifest digest where appropriate.
- Text is HTML-escaped. Reports have no scripts, remote assets or analytics;
  a restrictive content security policy permits only local embedded raster media.
- Optional media must have a digest, raster type and explicit classification
  (`measured_simulation_trace` or `simulation_preview`). A trace animation is
  never labeled real-camera footage.
- Recipe edits change its digest. Write a new report snapshot/revision; never
  rebind an old result silently. Snapshot directories cannot be overwritten.
- Adding a grasp/insertion recipe later can reuse the reporting mechanism, but
  requires its own supported executor, evidence, metrics and tests. This is an
  extension point, **not an implemented MVP2/MVP3 workflow**.

## Developer verification

```bash
pytest tests/test_reporting.py
pytest
```

The reporting tests cover actual five-call record integration, partial runs,
changed metrics, negative outcomes, zero baselines, missing facts, strict types,
recipe drift, record mutation, cross-run contamination, path/symlink escapes,
duplicate/nonfinite JSON, HTML escaping and preservation of existing bundles.
For browser checks, see [the Playwright harness](../examples/reporting/qa.mjs).
