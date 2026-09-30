"""Explicit read-only adapters; no fitting, simulation, or activation here."""

from __future__ import annotations

from pathlib import Path

from .report import ReportError, _json, digest, load_recipe, write_bundle


def export_toolkit_run(run_dir, recipe_path, destination, *, package_dir=None, session_path=None):
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

    checkpoint = None
    if session_path is not None:
        raw = Path(session_path).read_bytes()
        checkpoint = _json(raw)
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != "newton.guided-session/v1":
            raise ReportError("Expected a guided session snapshot")
        refs = checkpoint.get("artifacts", {})
        if not isinstance(refs, dict) or not all(isinstance(ref, dict) for ref in refs.values()):
            raise ReportError("Guided session artifact references must be objects")
        if not isinstance(checkpoint.get("events", []), list) or not all(
            isinstance(event, dict) for event in checkpoint.get("events", [])
        ):
            raise ReportError("Guided session events must be an array of records")
        for stage, record in {
            "analyze": "analysis",
            "intake": "guided_intake",
            "plan": "plan",
            "fit": "fit",
            "validate": "validation",
            "write": "manifest",
        }.items():
            if (stage in refs or record in records) and (
                record not in records or refs.get(stage, {}).get("sha256") != original_hashes.get(record)
            ):
                raise ReportError(f"Guided session does not match saved {stage} record")
        if (
            not analysis
            or not intake
            or (
                checkpoint.get("session_id") != intake.get("session_id")
                or checkpoint.get("revision") != intake.get("revision")
            )
        ):
            raise ReportError("Guided session revision does not match the analyzed inputs")
        records["guided_session"] = {**checkpoint, "run_id": run_id}
        view["source_sha256"]["guided_session"] = digest(raw)

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
    if checkpoint is not None:
        # Compute advice from the immutable snapshot, never from status.json or
        # expected recipe results. No readiness or activation decision is changed.
        from newton_calibration.guided.actions import build_action_plan

        try:
            actions = build_action_plan(checkpoint)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ReportError("Guided session advice inputs are malformed") from exc
        declared = {field["id"] for field in recipe["reporting"]["fields"]}
        if "action_plan" in declared:
            derived("action_plan", actions)
        derived("next_action", actions["summary"])
        events = [
            event for event in checkpoint.get("events", []) if event.get("revision") == checkpoint.get("revision")
        ]
        for stage in ("plan", "fit", "validate", "write"):
            attempted = any(str(event.get("event", "")).startswith(stage + ":") for event in events)
            if (
                events
                and not attempted
                and stage not in checkpoint.get("artifacts", {})
                and checkpoint.get("active_step") not in {stage, "plan_collection" if stage == "plan" else stage}
                and f"{stage}.status" not in facts
            ):
                derived(f"{stage}.status", "not_run")
                derived(f"{stage}.summary", "Not executed in this session revision; see next actions.")
        if not fit:
            derived(
                "discoveries",
                "Analysis completed. Remaining setup or evidence checks have owners and next actions below; screening alone does not prove parameter identifiability.",
            )
    records["view"] = view
    return write_bundle(
        recipe_path, destination, run_id=run_id, execution_kind="public_api", records=records, facts=facts
    )
