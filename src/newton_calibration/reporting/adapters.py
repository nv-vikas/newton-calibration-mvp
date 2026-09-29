"""Explicit read-only adapters; no fitting, simulation, or activation here."""

from __future__ import annotations

from pathlib import Path

from .report import ReportError, _json, digest, load_recipe, write_bundle


def export_toolkit_run(run_dir, recipe_path, destination, *, package_dir=None):
    """Snapshot available five-call records. Absent records remain unreported.

    status.json is deliberately not a source: it is a cosmetic progress surface.
    Package activation is copied only from a same-run manifest, never inferred
    from validation error or from a completed API call.
    """
    root = Path(run_dir)
    records, original_hashes = {}, {}
    for name in ("analysis", "plan", "fit", "validation", "guided_intake"):
        path = root / f"{name}.json"
        if path.is_file():
            raw = path.read_bytes()
            records[name] = _json(raw)
            original_hashes[name] = digest(raw)
    if not records:
        raise ReportError("No durable five-call records found")
    if package_dir is not None:
        raw = (Path(package_dir) / "manifest.json").read_bytes()
        records["manifest"] = _json(raw)
        original_hashes["manifest"] = digest(raw)
    ids = {record.get("run_id") for record in records.values()}
    if len(ids) != 1 or not all(isinstance(run_id, str) and run_id for run_id in ids):
        raise ReportError("Do not mix records from different runs")
    run_id = ids.pop()
    analysis = records.get("analysis", {})
    intake = records.get("guided_intake", {})
    plan = records.get("plan", {})
    fit = records.get("fit", {})
    validation = records.get("validation", {})
    manifest = records.get("manifest", {})
    if fit and plan and fit.get("plan") != plan:
        raise ReportError("Fit record does not use the saved plan")
    if validation and fit and validation.get("fit") != fit:
        raise ReportError("Validation record does not use the saved fit")
    recipe = load_recipe(recipe_path)
    if plan and plan.get("recipe") not in recipe["execution"]["supported_recipe_ids"]:
        raise ReportError("Execution recipe is not supported by this report adapter")
    env = plan.get("environment", analysis.get("environment", {}))
    if intake and intake.get("environment") != env:
        raise ReportError("Guided intake does not match the recorded environment")
    view = {
        "adapter": "five-call-report-adapter/v1",
        "run_id": run_id,
        "source_sha256": original_hashes,
        "scope": f"{env.get('robot_id', 'Articulation')} · free-motion response",
        "execution_note": "Built from saved toolkit result records. Report generation did not rerun physics or authorize hardware.",
        "limitations": "Free-motion metrics do not establish grasp, contact, insertion or real-task transfer. Review setup and parameter identification limits.",
        "next_action": "Review outstanding readiness and parameter checks, verify the package with its loader, then evaluate the intended task separately.",
        "metric_label": "held-out joint-position RMSE (episode-average)",
        "metric_unit": "rad",
    }
    if env.get("adapter") == "analytic":
        view["execution_note"] = (
            "Five-call records from the analytic test backend. No Newton GPU or real-robot validation was performed by this run."
        )
        view["limitations"] = (
            "Software integration result only; the analytic test backend cannot qualify a Newton calibration package or establish task transfer."
        )
    facts = {}

    def bind(key, source, pointer):
        facts[key] = (source, pointer)

    def derived(key, value):
        view[key] = value
        bind(key, "view", "/" + key)

    for key in tuple(view):
        if key not in {"adapter", "run_id", "source_sha256"}:
            bind(key, "view", "/" + key)
    if analysis:
        derived("inputs", f"Robot asset + recorded evidence; {len(analysis.get('joints', []))} measured joints.")
        derived(
            "discoveries",
            "; ".join(analysis.get("warnings", [])) or "No warnings recorded by analyze; see readiness checks.",
        )
        bind("readiness", "analysis", "/readiness")
        derived(
            "controller_tool",
            {
                "environment": env,
                "tool_confirmation": "Not separately attested by this reporting adapter; inspect source profile.",
            },
        )
        derived(
            "analyze.status",
            "completed",
        )
        derived("analyze.summary", "Evidence inspected. Readiness checks and unresolved inputs are recorded below.")
        if intake:
            declared = intake.get("inputs", {})
            derived(
                "controller_tool",
                {
                    "controller": declared.get("controller"),
                    "tool": declared.get("tool"),
                    "confirmations": intake.get("confirmations", {}),
                    "source": "Recorded user/engineer declarations; not independent hardware verification",
                },
            )
        derived(
            "uncertainty",
            {
                "identifiability": analysis.get("identifiability", {}),
                "confidence_intervals": "Not reported by this adapter.",
            },
        )
    if plan:
        derived("plan.status", "recorded")
        derived("plan.summary", f"{len(plan.get('parameters', []))} parameter settings; split and runtime recorded.")
        derived(
            "parameters",
            {"specifications": plan.get("parameters", []), "selected": fit.get("best", {}).get("parameters", {})},
        )
        bind("objective", "plan", "/objective_weights")
        bind("runtime", "plan", "/environment")
        derived("split", {"train": plan.get("train_episodes", []), "heldout": plan.get("heldout_episodes", [])})
        derived(
            "collection",
            {
                "train": plan.get("train_episodes", []),
                "heldout": plan.get("heldout_episodes", []),
                "generated_motions": "Not established by these records",
                "preview": "Not established by these records",
            },
        )
        derived(
            "gates", {"declared": plan.get("validation_gates", {}), "recorded_results": validation.get("gates", {})}
        )
    if fit:
        derived("fit.status", "recorded")
        derived(
            "fit.summary",
            f"{fit.get('completed_generations', 'Unreported')} completed generations; candidate history preserved by the run.",
        )
        bind("optimizer", "fit", "/optimizer")
    if validation:
        derived("validate.status", "completed" if validation.get("passed") is True else "failed")
        derived(
            "validate.summary",
            "Baseline and selected settings compared on held-out recordings; consult gates and regressions.",
        )
        bind("baseline_error", "validation", "/baseline_metrics/position_rmse_rad")
        bind("tuned_error", "validation", "/calibrated_metrics/position_rmse_rad")
        bind("validation_passed", "validation", "/passed")
    if manifest:
        derived("write.status", "recorded")
        derived("write.summary", "Package manifest saved. Activation decision is separate from file creation.")
        bind("activation_allowed", "manifest", "/activation_allowed")
        derived(
            "outputs",
            {"manifest": manifest, "notice": "This report is not a substitute for package-loader verification."},
        )
    records["view"] = view
    return write_bundle(
        recipe_path, destination, run_id=run_id, execution_kind="public_api", records=records, facts=facts
    )
