"""Deterministic next-action advice. Never authorizes fitting or hardware."""

from __future__ import annotations


def build_action_plan(session):
    """Translate recorded questions into owners, deliverables and completion checks.

    These are recommendations, not execution results. The existing readiness
    checks remain authoritative. Do not infer that screening proves identifiable
    parameters, or request a new capture just because replay software is missing.
    """
    state = session.get("state", "needs_information")
    questions = session.get("questions", [])
    keys = {q["key"] for q in questions}
    inputs = session.get("inputs", {})
    needs = session.get("evidence_needs") or {}
    items = []

    def add(key, owner, action, done, sources=(), depends=()):
        items.append(
            {
                "id": key,
                "priority": len(items) + 1,
                "owner": owner,
                "action": action,
                "done_when": done,
                "source_questions": sorted(set(sources) & keys),
                "depends_on": list(depends),
                "status": "proposed",
            }
        )

    data_status = "not_assessed"
    data_message = "Inspect existing evidence before deciding what additional data is needed."
    if session.get("error") or state == "needs_attention":
        add(
            "recover",
            "Toolkit engineering / agent",
            "Resolve the recorded execution or input error.",
            "The cause is resolved and the same revision resumes; changed inputs create a new revision. No fallback fit.",
            {"inputs"},
        )
        data_status, data_message = "deferred", "Resolve the execution issue before requesting more robot data."
    elif state == "choose_recipe":
        add(
            "choose_recipe",
            "User, guided by agent",
            "Choose an available calibration recipe.",
            "The user selects a supported recipe; future grasp/insertion recipes are not executed.",
            {"recipe"},
        )
    elif state == "completed" or "write" in session.get("artifacts", {}):
        add(
            "review_result",
            "User / robot engineer",
            "Review the validation decision and package restrictions.",
            "The scope, failed checks and activation decision are understood; any use goes through the verified package loader.",
        )
        data_status, data_message = (
            "not_requested",
            "No automatic recollection request; use validation findings to decide the next experiment.",
        )
    elif state in {"ready_to_fit", "ready_to_resume"} and session.get("fit_allowed"):
        add(
            "authorize_fit",
            "User",
            "Review the locked plan and explicitly start or resume calibration.",
            "The user requests execution. This permits simulation fitting, not real robot motion.",
        )
        data_status, data_message = "not_requested", "No additional data requested for the current fitting scope."
    elif state in {"analyzing", "planning", "fitting", "validating", "packaging", "preparing_collection"}:
        add(
            "monitor",
            "Agent",
            "Monitor the current step and its recorded result.",
            "The step completes or produces a specific error; no parallel writer or bypass is started.",
        )
    else:
        if "asset" in keys or "inputs" in keys:
            add(
                "repair_inputs",
                "Agent / asset owner",
                "Resolve the asset or changed-input issue.",
                "The USD can be inspected and the current input revision is explicitly recorded.",
                {"asset", "inputs"},
            )
        replay_keys = keys & {"controller_rate", "controller_support", "controller_timing"}
        if replay_keys:
            add(
                "qualify_replay",
                "Toolkit engineering / agent",
                "Fix or qualify the controller-replay adapter.",
                "Logged commands retain their original timing; supported filtering/limiting and physics substeps pass replay tests. Do not change the declared physics step merely to hide a rate mismatch.",
                replay_keys,
            )
        if "controller_processing" in keys:
            add(
                "qualify_command_processing",
                "Toolkit engineering / agent",
                "Resolve requested versus driver-published commands.",
                "Use source-backed published targets, or qualify the recorded limiter/filter replay. Existing requested targets are not relabeled as what the robot received.",
                {"controller_processing"},
            )
        if state == "needs_evidence_description" or "evidence" in keys:
            add(
                "interpret_evidence",
                "Agent, with data owner",
                "Map and describe the recordings already supplied.",
                "Source signals, clocks, units, joint bindings and independent capture IDs are inspectable; existing data is not silently discarded.",
                {"evidence", "joint_bindings"},
            )
        profile_keys = keys & {"controller", "environment", "confirm.bounds"}
        if profile_keys:
            add(
                "prepare_profile",
                "Agent",
                "Prepare a short simulation-configuration review sheet.",
                "Proposed gains, limits, bounds and controller assumptions cite USD/config/metadata sources. Unknowns remain explicit, not confirmed defaults.",
                profile_keys,
            )
        confirm_keys = {key for key in keys if key.startswith("confirm.")} | (keys & {"tool", "controller"})
        if confirm_keys:
            add(
                "confirm_setup",
                "Robot engineer, guided by agent",
                "Confirm the proposed setup against the real robot.",
                "Only unresolved joint directions/zeros, controller settings and tool/payload facts are reviewed; each confirmation names its verifier and source.",
                confirm_keys,
                ("prepare_profile",) if profile_keys else (),
            )
        missing_data = (
            not inputs.get("evidence")
            or needs.get("heldout_needed") is True
            or any(row.get("disposition") == "collect" for row in needs.get("parameters", []))
        )
        setup_open = bool(keys - {"collection", "readiness"})
        if setup_open:
            data_status = "deferred"
            data_message = "No new robot data requested yet. Resolve replay/setup or evidence interpretation first; reuse the supplied recordings."
            if not inputs.get("evidence"):
                data_message = (
                    "Evidence is missing, but collection waits for the supported controller and reviewed setup."
                )
        elif missing_data:
            data_status = "needed"
            data_message = "Targeted evidence is needed. Review motion/preview artifacts with the operator before any approved real collection."
        elif session.get("analysis_status") == "completed":
            data_status, data_message = (
                "not_requested",
                "No new robot data requested for the current scope; evidence screening is not proof of identifiability.",
            )
        if missing_data:
            add(
                "prepare_collection",
                "Agent / robot engineer",
                "Prepare or review collection only for the remaining evidence gaps.",
                "Supported motions, required signals, independent holdouts and preview status are recorded. The operator approves any hardware run separately.",
                keys & {"collection"},
                tuple(item["id"] for item in items),
            )
        elif not items and "readiness" in keys:
            add(
                "resolve_readiness",
                "Agent / robot engineer",
                "Resolve the remaining analysis checks.",
                "Every failed readiness item has a source-backed correction; none is overridden.",
                {"readiness"},
            )
        add(
            "recheck",
            "Toolkit",
            "Re-run analysis with the reviewed answers and existing recordings.",
            "Produce a locked fitting plan only when all gates pass; otherwise request specific missing evidence, not blanket recollection.",
            {"readiness"},
            tuple(item["id"] for item in items),
        )
    return {
        "schema": "newton.guided-actions/v1",
        "basis_revision": session.get("revision"),
        "basis_state": state,
        "analysis_status": session.get("analysis_status", "not_run"),
        "summary": items[0]["action"],
        "items": items,
        "data_collection": {"status": data_status, "message": data_message},
        "notice": "Proposed next actions, not completed work or hardware authorization.",
    }
