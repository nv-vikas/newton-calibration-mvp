"""One guided path through the existing five calls; no exploratory fallback."""

from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path

from newton_calibration.collection import CalibrationRequest, CollectionPlan, MotionSpec
from newton_calibration.core.io import sha256_file, utc_now, write_json
from newton_calibration.core.models import jsonable
from newton_calibration.isaaclab import tuning
from newton_calibration.reporting import build_report, render_report
from newton_calibration.reporting.adapters import export_toolkit_run

from . import codec
from .catalog import ARM_ID, ARM_RECIPE, get_recipe, list_recipes, question
from .intake import CONFIRMATIONS, SECTIONS, confirmation_basis, inspect_inputs
from .store import artifact, fingerprint, locked, read, remember, save


def start(*, asset, goal, directory, recipe=None, evidence=None):
    """Start with what the user has. No physics, confirmations or robot actions."""
    asset = Path(asset).expanduser().resolve()
    if not asset.is_file() or not isinstance(goal, str) or not goal.strip():
        raise ValueError("Provide an existing USD and a task goal")
    if recipe is not None:
        get_recipe(recipe)
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    session = {
        "schema": "newton.guided-session/v1",
        "session_id": uuid.uuid4().hex,
        "created_at": utc_now(),
        "revision": 1,
        "asset": str(asset),
        "asset_sha256": sha256_file(asset),
        "goal": goal,
        "recipe_id": recipe,
        "recipe_sha256": get_recipe(recipe)["_sha256"] if recipe else None,
        "inputs": {"evidence": str(Path(evidence).expanduser().resolve()) if evidence else None},
        "confirmations": {},
        "state": "choose_recipe" if recipe is None else "needs_information",
        "artifacts": {},
        "history": [],
        "events": [],
        "questions": [],
        "proposals": {},
        "real_robot_commands_sent": False,
        "hardware_execution_authorized": False,
    }
    save(root, session, "session_started")
    return review(root)


def provide(directory, answers, *, source, confirmed_by=None):
    """Record answers/provenance; explicit confirmations are tied to setup bytes.

    An agent may submit user-confirmed facts, but must never populate confirmed_by
    from its own guesses. This is an assertion record, not identity authentication.
    """
    if not isinstance(answers, dict) or not isinstance(source, str) or not source.strip():
        raise ValueError("Answers must be an object with a source")
    if set(answers) - (SECTIONS | {"recipe", "confirm"}):
        raise ValueError(f"Unknown answer keys: {sorted(set(answers) - (SECTIONS | {'recipe', 'confirm'}))}")
    names = answers.get("confirm", [])
    if not isinstance(names, list) or any(name not in CONFIRMATIONS for name in names):
        raise ValueError("confirm must list mapping, controller, tool and/or bounds")
    if names and (not isinstance(confirmed_by, str) or not confirmed_by.strip()):
        raise ValueError("Explicit confirmations require the name of the person who verified them")
    with locked(directory) as (root, session):
        previous = deepcopy(session)
        if "recipe" in answers:
            recipe = get_recipe(answers["recipe"])
            session.update(recipe_id=recipe["id"], recipe_sha256=recipe["_sha256"])
        for key in SECTIONS & answers.keys():
            value = deepcopy(answers[key])
            if key not in {"evidence", "joint_bindings"} and value is not None and not isinstance(value, dict):
                raise ValueError(f"{key} must be an object, or null for unknown")
            if key == "environment" and value:
                # User/agent JSON cannot smuggle booleans around the confirmation ledger.
                for flag in ("profile_confirmed", "controller_profile_confirmed", "controller_profile_source"):
                    value.pop(flag, None)
                if value.get("asset_path") and Path(value["asset_path"]).expanduser().resolve() != Path(
                    session["asset"]
                ):
                    raise ValueError("An environment for another USD needs a new session")
            if key == "evidence" and isinstance(value, str):
                value = str(Path(value).expanduser().resolve())
            session["inputs"][key] = value
        current_asset = sha256_file(session["asset"])
        session["asset_sha256"] = current_asset
        for name in names:
            session["confirmations"][name] = {
                "by": confirmed_by,
                "source": source,
                "at": utc_now(),
                "basis": confirmation_basis(session["inputs"], name, current_asset),
            }
        # Validate before publishing a revision. Unknown/null values are allowed.
        if session["recipe_id"]:
            inspect_inputs(session)
        revision = session["revision"]
        write_json(root / "revisions" / f"{revision:04d}.json", previous)
        session["history"].append(
            {"revision": revision, "state": previous["state"], "artifacts": previous["artifacts"]}
        )
        session.update(revision=revision + 1, state="needs_information", artifacts={}, questions=[], proposals={})
        for field in (
            "observed_input_fingerprint",
            "error",
            "collection",
            "customer_report",
            "activation_allowed",
            "validation_passed",
            "fit_allowed",
            "next_action",
            "analysis_status",
            "fit_readiness",
            "evidence_needs",
            "active_step",
        ):
            session.pop(field, None)
        session["last_answers"] = {"source": source, "keys": sorted(answers), "confirmed_by": confirmed_by}
        save(root, session, "inputs_updated; prior results retained but no longer current")
    return review(directory)


def _intake(root, session):
    if sha256_file(session["asset"]) != session["asset_sha256"]:
        raise ValueError("USD changed. Submit reviewed inputs to start a new revision; old results cannot be reused")
    if get_recipe(session["recipe_id"])["_sha256"] != session["recipe_sha256"]:
        raise ValueError("Recipe changed. Re-select the recipe to create a new revision")
    intake = inspect_inputs(session)
    session["questions"] = intake["questions"]
    session["proposals"] = intake["proposals"]
    session["usd_inspection"] = intake["usd"]
    session["scope"] = list(intake["environment"].tuning_targets) if intake["environment"] else []
    session["scope_notice"] = get_recipe(session["recipe_id"])["guided"]["default_scope_notice"]
    return intake


def _public(session):
    return deepcopy(session)


def status(directory):
    """Read the last atomic snapshot, even while a long fit owns the writer lock.

    This reports recorded progress; it does not re-inspect changing inputs.
    Use review/advance when idle to re-check readiness.
    """
    session = read(Path(directory).expanduser().resolve() / "session.json")
    if session.get("schema") != "newton.guided-session/v1":
        raise ValueError("Unsupported session schema")
    return _public(session)


def review(directory):
    """Get the next questions. Does not execute scientific calls."""
    with locked(directory) as (root, session):
        if session["recipe_id"] is None:
            session["questions"] = [
                question(
                    "recipe",
                    "Which calibration recipe would you like to run?",
                    "Arm joint tuning is available; grasp and insertion are planned.",
                    blocks="all",
                    example=ARM_ID,
                )
            ]
            session["recipes"] = list_recipes()
        else:
            try:
                _intake(root, session)
            except (ValueError, OSError, TypeError) as exc:
                session["state"] = "needs_information"
                session["fit_allowed"] = False
                session["activation_allowed"] = False
                session["questions"] = [
                    question("inputs", "Review the changed or invalid input.", str(exc), blocks="all")
                ]
        save(root, session, "reviewed")
        return _public(session)


def advance(directory, *, execute=False, preview=None, design_probe=None):
    """Advance until the next human boundary. `execute=True` permits fitting.

    Preview/probe hooks are trusted installed code, supplied explicitly by the
    application—not loaded from recipe, evidence, or answers. There is no robot
    driver hook, bypass flag or exploratory mode.
    """
    with locked(directory) as (root, session):
        if session["recipe_id"] is None:
            session["state"] = "choose_recipe"
            save(root, session, "recipe_selection_required")
            return _public(session)
        try:
            return _advance(root, session, execute, preview, design_probe)
        except Exception as exc:
            session["state"] = "needs_attention"
            session["fit_allowed"] = False
            session["activation_allowed"] = False
            session["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "step": session.get("active_step"),
                "recovery": "Resolve the reported issue; advance again resumes the same revision. Changed inputs require provide().",
            }
            save(root, session, "execution_failed; no fallback attempted")
            raise


def _advance(root, session, execute, preview, design_probe):
    intake = _intake(root, session)
    session["fit_allowed"] = False
    env = intake["environment"]
    if env is None or any(q["blocks"] == "all" for q in session["questions"]):
        session["state"] = "needs_information"
        save(root, session, "setup_action_required")
        return _public(session)
    if intake.get("evidence_issue"):
        session["state"] = "needs_evidence_description"
        save(root, session, "existing_evidence_preserved; interpretation_required")
        return _public(session)
    inputs = session["inputs"]
    evidence = intake["evidence"]
    input_fingerprint = fingerprint(
        {
            "environment": jsonable(env),
            "inputs": inputs,
            "evidence": evidence.spec.fingerprint if evidence else None,
            "recipe": session["recipe_sha256"],
        }
    )
    previous = session.get("observed_input_fingerprint")
    if previous and previous != input_fingerprint:
        raise ValueError("Input files changed after analysis; provide reviewed inputs to create a new revision")
    session["observed_input_fingerprint"] = input_fingerprint
    request_data = dict(inputs.get("request") or {})
    request_data["target_parameters"] = tuple(env.tuning_targets)
    request = CalibrationRequest(**request_data)
    attempt = root / "attempts" / f"revision-{session['revision']:04d}"
    if "analyze" not in session["artifacts"]:
        session.update(state="analyzing", active_step="analyze")
        save(root, session, "analyze:started")
        analysis = tuning.analyze(
            env=env,
            evidence=evidence,
            request=request,
            recipe=get_recipe(session["recipe_id"])["guided"]["executor"],
            workdir=attempt / "runs",
        )
        remember(root, session, "analyze", Path(analysis.workdir) / "analysis.json")
    else:
        analysis = codec.analysis(artifact(root, session, "analyze"))
    if "intake" not in session["artifacts"]:
        definition = get_recipe(session["recipe_id"])
        context = {
            "schema": "newton.guided-intake/v1",
            "run_id": analysis.run_id,
            "session_id": session["session_id"],
            "revision": session["revision"],
            "goal": session["goal"],
            "environment": jsonable(env),
            "inputs": inputs,
            "confirmations": session["confirmations"],
            "recipe_sha256": session["recipe_sha256"],
            "recipe_definition": {key: value for key, value in definition.items() if not key.startswith("_")},
            "input_fingerprint": input_fingerprint,
            "last_answers": session.get("last_answers"),
            "earlier_revisions": session["history"],
            "hardware_execution_authorized": False,
        }
        remember(root, session, "intake", write_json(Path(analysis.workdir) / "guided_intake.json", context))
    else:
        artifact(root, session, "intake")
    session["analysis_status"] = "completed"
    session["fit_readiness"] = analysis.readiness
    session["evidence_needs"] = analysis.evidence_needs
    collection_needed = (
        evidence is None
        or analysis.evidence_needs["heldout_needed"]
        or any(row["disposition"] == "collect" for row in analysis.evidence_needs["parameters"])
    )
    if collection_needed:
        return _collect(root, session, intake, analysis, preview, design_probe)
    if session["questions"] or not all(analysis.readiness.values()):
        session["state"] = "needs_information"
        if not all(analysis.readiness.values()):
            failed = [name for name, ok in analysis.readiness.items() if not ok]
            session["questions"].append(
                question(
                    "readiness",
                    "Resolve these fitting requirements.",
                    "; ".join(failed) + ". See analysis.json for evidence and suggested actions.",
                )
            )
        save(root, session, "analysis_completed; fitting_not_ready")
        return _public(session)
    if "plan" not in session["artifacts"]:
        session.update(state="planning", active_step="plan")
        save(root, session, "plan:started")
        plan = tuning.plan(analysis, intent="fit")  # keeps EVERY native readiness gate
        remember(root, session, "plan", Path(plan.workdir) / "plan.json")
    else:
        plan = codec.plan(artifact(root, session, "plan"))
    session["fit_allowed"] = True
    if not execute:
        if all(key in session["artifacts"] for key in ("fit", "validate", "write", "report")):
            for key in session["artifacts"]:
                artifact(root, session, key)
            session["state"] = "completed"
            save(root, session, "completed_records_verified; no_execution_requested")
            return _public(session)
        session["state"] = "ready_to_resume" if "fit" in session["artifacts"] else "ready_to_fit"
        save(root, session, "fit_requires_explicit_execution_request")
        return _public(session)
    budget = inputs.get("fit_budget") or {}
    if set(budget) - {"generations", "population"}:
        raise ValueError("Fit budget accepts generations and population only")
    if "fit" not in session["artifacts"]:
        session.update(state="fitting", active_step="fit")
        save(root, session, "fit:started_or_resumed")
        fit = tuning.fit(plan, resume=True, **budget)
        remember(root, session, "fit", Path(plan.workdir) / "fit.json")
    else:
        fit = codec.fit(artifact(root, session, "fit"))
    if "validate" not in session["artifacts"]:
        session.update(state="validating", active_step="validate")
        save(root, session, "validate:started")
        validation = tuning.validate(fit)
        remember(root, session, "validate", Path(plan.workdir) / "validation.json")
    else:
        validation = codec.validation(artifact(root, session, "validate"))
    if "write" not in session["artifacts"]:
        session.update(state="packaging", active_step="write")
        save(root, session, "write:started")
        # Each write attempt gets a new directory. A failed/partial package is
        # preserved, not mistaken for a completed write on resume.
        package_dir = attempt / "packages" / uuid.uuid4().hex[:12]
        package = tuning.write(validation, output=package_dir)
        remember(root, session, "write", package.manifest_path)
    manifest = artifact(root, session, "write")
    package_dir = (root / session["artifacts"]["write"]["path"]).parent
    if "report" not in session["artifacts"]:
        report_dir = attempt / "reports" / uuid.uuid4().hex[:12]
        bundle = export_toolkit_run(plan.workdir, ARM_RECIPE, report_dir, package_dir=package_dir)
        model = build_report(ARM_RECIPE, bundle)
        render_report(model, report_dir / "report.html")
        remember(root, session, "report", bundle)
    report_path = (root / session["artifacts"]["report"]["path"]).parent / "report.html"
    session["customer_report"] = str(report_path)
    session.update(
        state="completed",
        active_step=None,
        validation_passed=validation.passed,
        activation_allowed=manifest.get("activation_allowed", False),
    )
    session.pop("error", None)
    save(root, session, "workflow_complete; scientific_and_activation_decisions_retained")
    return _public(session)


def _collect(root, session, intake, analysis, preview, design_probe):
    session["fit_allowed"] = False
    if not intake["motion_supported"]:
        session["state"] = "needs_collection_setup"
        save(root, session, "analysis_completed; identify_simulation_controller_before_motion_generation")
        return _public(session)
    motion_data = session["inputs"].get("collection")
    if not motion_data:
        session["state"] = "needs_collection_setup"
        session["questions"].append(
            question(
                "collection",
                "Bind the Isaac Lab scene and a reviewed simulation motion envelope.",
                "Need starting pose, controlled joints, limits, amplitude/rate caps, scene ID and provenance. Do not infer safe hardware limits from USD.",
                blocks="collection",
            )
        )
        save(root, session, "analysis_completed; collection_scene_required")
        return _public(session)
    motion = MotionSpec(**motion_data)
    rate = (session["inputs"].get("controller") or {}).get("command_rate_hz")
    if rate != motion.command_rate_hz:
        raise ValueError("Collection command rate must match the declared controller rate")
    if "collection" in session["artifacts"]:
        result = artifact(root, session, "collection")
        needs_probe = result.get("design", {}).get("adaptive_search", {}).get("status") == "needs_dynamics_probe"
        if (result["preview"]["status"] != "complete" and preview is not None) or (
            needs_probe and design_probe is not None
        ):
            # A retry has its own analysis and artifacts. Never rewrite an old
            # collection record or erase the failed/pending preview.
            session["history"].append(
                {"revision": session["revision"], "collection_retry": deepcopy(session["artifacts"])}
            )
            session["artifacts"].pop("collection")
            session["artifacts"].pop("analyze")
            session["artifacts"].pop("intake", None)
            save(root, session, "collection_retry_requested")
            return _advance(root, session, False, preview, design_probe)
    else:
        session.update(state="preparing_collection", active_step="plan_collection")
        save(root, session, "plan:collection_started")
        plan = tuning.plan(
            analysis, intent="collect", collection=motion, preview=preview, design_probe=design_probe, video=True
        )
        if not isinstance(plan, CollectionPlan):
            raise TypeError("Collection branch must return CollectionPlan")
        remember(root, session, "collection", Path(plan.workdir) / "collection_plan.json")
        result = jsonable(plan)
    session["collection"] = result
    preview_record = result["preview"]
    design = result.get("design", {}).get("adaptive_search", {})
    reviewed_design = design.get("status") in {
        "recipe_only_explicit",
        "predicted_coverage_reached",
        "candidate_catalog_exhausted",
        "probe_budget_reached",
        "selection_budget_reached",
        "catalog_visited_with_failures",
    }
    if (
        result.get("status") == "preview_complete_review_required"
        and preview_record.get("status") == "complete"
        and preview_record.get("screen_passed") is True
        and reviewed_design
    ):
        session["state"] = "awaiting_operator_review_and_real_data"
    else:
        session["state"] = "collection_needs_review"
    session["next_action"] = (
        "Review motion files, simulation screen/video, remaining evidence gaps and real-robot limits with the operator. Run approved collection separately, then attach real logs to this session."
    )
    session["active_step"] = None
    save(root, session, "analysis_completed; collection_prepared_not_calibration")
    return _public(session)
