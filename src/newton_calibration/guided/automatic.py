"""Run supported work to a genuine input boundary, without a human to-do loop.

No LLM, robot connection, arbitrary recipe code, or readiness override. A trusted
host application may provide a resolver for further agent-owned preparation.
"""

from copy import deepcopy

from .actions import user_requests
from .preflight import BASELINE_FIELDS, inspect_simulation_profile
from .store import fingerprint, locked, save
from .workflow import advance, provide, review

_PREPARATION_KEYS = {"environment", "controller", "tool", "joint_bindings", "evidence", "collection", "confirm"}


def _prepare(state, simulation_profile):
    if state.get("workflow_mode") == "collection_only":
        return None
    if not state.get("recipe_id") or state.get("error") or "fit" in state.get("artifacts", {}):
        return None
    answers, sources, completed = {}, [], []
    inputs = state["inputs"]
    env = deepcopy(inputs.get("environment") or {})
    keys = {q["key"] for q in state.get("questions", [])}
    if simulation_profile and "environment" in keys and env.get("joint_map"):
        proposal = inspect_simulation_profile(simulation_profile, env["joint_map"])
        for field in BASELINE_FIELDS:
            # Existing partial declarations are not overwritten by an old profile.
            if not env.get(field):
                env[field] = proposal["environment_patch"][field]
        if env != inputs.get("environment"):
            answers["environment"] = env
            sources.append(f"Simulation profile {proposal['source']['path']} sha256:{proposal['source']['sha256']}")
            completed.append("loaded_simulation_baselines")
    # This is a supported simulation choice, never discovery of real firmware.
    controller = deepcopy(inputs.get("controller") or {})
    if (
        controller.get("simulation_mode") in (None, "", "unknown")
        and env.get("adapter") in {"analytic", "isaaclab_newton"}
        and controller.get("real_mode") in {"joint_position", "joint_impedance"}
        and controller.get("command_semantics") == "absolute_joint_position"
    ):
        controller["simulation_mode"] = "joint_position_pd"
        answers["controller"] = controller
        sources.append("Installed arm-joint recipe and explicit-PD runtime; real controller facts unchanged")
        completed.append("selected_supported_simulation_controller")
    # Review after other changes so bounds are attached to the new setup bytes.
    if not answers and "confirm.bounds" in keys and state.get("proposals", {}).get("parameters", {}).get("value"):
        answers["confirm"] = ["bounds"]
        sources.append(
            f"Simulation-only bounds from declared inputs/recipe sha256:{state['recipe_sha256']}; not hardware limits"
        )
        completed.append("reviewed_simulation_search_bounds")
    return {"answers": answers, "source": "; ".join(sources), "actions": completed} if answers else None


def run(
    directory,
    *,
    execute=False,
    simulation_profile=None,
    resolver=None,
    preview=None,
    design_probe=None,
    max_passes=6,
    agent_name="Calibration automation",
    expected_execution_basis=None,
):
    """Do supported preparation and all authorized calls; stop only at a boundary.

    ``execute=True`` is one authorization for this invocation's simulation fit,
    validation and packaging. Hardware remains outside this API. ``resolver`` is
    trusted installed code: it receives a snapshot and returns data-only
    {answers, source}, or None. Input files cannot select or execute a resolver.
    """
    if type(execute) is not bool:
        raise ValueError("execute must be an explicit boolean authorization")
    if type(max_passes) is not int or not 1 <= max_passes <= 32:
        raise ValueError("max_passes must be an integer from 1 to 32")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("An actual automation/agent identity is required")
    if resolver is not None and not callable(resolver):
        raise TypeError("resolver must be an explicitly trusted callable")
    completed, seen = [], set()
    state = review(directory)
    reason = "pass_limit"
    for _ in range(max_passes):
        state_key = fingerprint({"inputs": state["inputs"], "confirmations": state["confirmations"]})
        if state_key in seen:
            reason = "no_progress"
            break
        seen.add(state_key)
        update = _prepare(state, simulation_profile)
        if update is None:
            state = advance(
                directory,
                execute=execute,
                preview=preview,
                design_probe=design_probe,
                expected_execution_basis=expected_execution_basis,
            )
            if state["state"] in {
                "completed",
                "ready_to_fit",
                "ready_to_resume",
                "awaiting_operator_review_and_real_data",
                "collection_spec_prepared",
            }:
                reason = "workflow_boundary"
                break
            if resolver is not None:
                update = resolver(deepcopy(state))
        if update is None:
            reason = "source_or_capability_boundary"
            break
        if not isinstance(update, dict) or not isinstance(update.get("answers"), dict) or not update.get("source"):
            raise ValueError("Resolver must return data-only answers with provenance, or None")
        answers = update["answers"]
        if set(answers) - _PREPARATION_KEYS:
            raise ValueError("Automatic preparation cannot change recipe, task scope, or fit budget")
        # A trusted resolver supplies facts, not an alternate run request. Keep
        # the selected backend and tuning scope; another USD requires a session.
        env_update = answers.get("environment")
        if "environment" in answers and not isinstance(env_update, dict):
            raise ValueError("Automatic preparation cannot erase the declared environment")
        if isinstance(env_update, dict):
            for key in ("adapter", "tuning_targets", "parameter_specs"):
                previous = (state["inputs"].get("environment") or {}).get(key)
                if env_update.get(key) != previous:
                    raise ValueError(f"Automatic preparation cannot change declared environment.{key}")
        if not answers or all(state["inputs"].get(k) == v for k, v in answers.items()):
            reason = "no_progress"
            break
        prior_revision = state["revision"]
        state = provide(
            directory,
            answers,
            source=update["source"],
            reviewer_kind="agent",
            confirmed_by=agent_name,
            expected_revision=prior_revision,
        )
        completed.append(
            {
                "from_revision": prior_revision,
                "to_revision": state["revision"],
                "answer_keys": sorted(answers),
                "source": update["source"],
                "actions": update.get("actions", ["trusted_resolver_preparation"]),
            }
        )
    else:
        # Still perform analysis of the last prepared revision; never declare an
        # unanalysed setup complete just because the bounded preparation loop ends.
        state = advance(
            directory,
            execute=execute,
            preview=preview,
            design_probe=design_probe,
            expected_execution_basis=expected_execution_basis,
        )
    requests = user_requests(state)
    outcome = "completed" if state["state"] == "completed" else "awaiting_user" if requests else "toolkit_attention"
    with locked(directory) as (root, current):
        if any(current.get(key) != state.get(key) for key in ("revision", "state", "artifacts")):
            raise RuntimeError("Session changed while automatic continuation was running")
        current["continuation"] = {
            "schema": "newton.automatic-continuation/v1",
            "basis_revision": state["revision"],
            "basis_state": state["state"],
            "outcome": outcome,
            "reason": reason,
            "execution_requested": execute,
            "automatic_preparation": completed,
            "user_requests": requests,
            "hardware_execution_authorized": False,
            "notice": "Toolkit attention is an unresolved software/source issue, not a completed or running job.",
        }
        current.setdefault("continuation_history", []).append(deepcopy(current["continuation"]))
        save(root, current, f"automatic_continuation:{outcome}")
        return deepcopy(current)
