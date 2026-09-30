"""Human-readable CLI and JSON surface for an agent. No LLM dependency."""

import importlib
import json
import uuid
from pathlib import Path

from .catalog import ARM_ID, list_recipes
from .store import read
from .workflow import advance, provide, review, start, status


def add_parser(subparsers):
    parser = subparsers.add_parser("guide", help="Guided recipe selection, intake, collection and calibration")
    actions = parser.add_subparsers(dest="guide_action")
    recipes = actions.add_parser("recipes", help="List available and future recipes")
    recipes.add_argument("--json", action="store_true")
    begin = actions.add_parser("start", help="Start with a USD and task goal; other inputs may be unknown")
    begin.add_argument("--asset", required=True)
    begin.add_argument("--goal", required=True)
    begin.add_argument("--recipe")
    begin.add_argument("--evidence", help="Existing bound-evidence or intake descriptor JSON")
    begin.add_argument("--session", required=True)
    begin.add_argument("--json", action="store_true")
    for action in ("review", "status", "provide", "advance", "wizard"):
        sub = actions.add_parser(action)
        sub.add_argument("--session", required=True)
        sub.add_argument("--json", action="store_true")
        if action == "provide":
            sub.add_argument("--answers", required=True, help="Data-only JSON with answers; null means unknown")
            sub.add_argument("--source", required=True)
            sub.add_argument("--confirmed-by", help="Person who actually verified the explicitly listed confirmations")
        if action == "advance":
            sub.add_argument("--execute", action="store_true", help="Permit fitting once every readiness check passes")
            sub.add_argument(
                "--preview-factory",
                help="Explicitly trusted local module:factory; no factory from input JSON is executed",
            )
            sub.add_argument(
                "--design-probe-factory", help="Trusted local module:factory using the same initialized scene"
            )
    return parser


def _factory(value):
    module, colon, name = value.partition(":")
    if not colon or not module or not name:
        raise ValueError("Factory must be trusted module:factory")
    return getattr(importlib.import_module(module), name)()


def describe(state):
    lines = [
        f"Session: {state['session_id']} · revision {state['revision']}",
        f"Next: {state['state'].replace('_', ' ')}",
    ]
    if state.get("analysis_status"):
        lines.append("Analysis: completed. This does not mean fitting is ready.")
    if state.get("scope"):
        lines.append("Requested parameters: " + ", ".join(state["scope"]))
    if state.get("action_plan"):
        actions = state["action_plan"]
        lines.append("\nWhat to do next:")
        for item in actions["items"]:
            lines.extend(
                [f"{item['priority']}. {item['action']} — {item['owner']}", f"   Done when: {item['done_when']}"]
            )
        lines.append("\nData collection: " + actions["data_collection"]["message"])
        lines.append("Detailed input questions remain available with --json or guide wizard.")
    else:
        for item in state.get("questions", []):
            lines.extend([f"\n• {item['question']} [{item['key']}]", "  " + item["why"]])
    if state.get("collection"):
        result = state["collection"]
        lines.extend(
            [
                f"\nCollection: {len(result['episodes'])} motion files; preview {result['preview']['status']}.",
                f"Files: {result['workdir']}",
                "Simulation proposals only. Operator approval is still required.",
            ]
        )
    if state.get("customer_report"):
        lines.extend(
            [
                f"\nReport: {state['customer_report']}",
                f"Held-out validation: {state.get('validation_passed')}; package activation: {state.get('activation_allowed')}",
            ]
        )
    if state.get("next_action"):
        lines.append(state["next_action"])
    if state.get("error"):
        lines.append("Attention: " + state["error"]["message"])
    lines.append("No commands have been sent to a real robot.")
    return "\n".join(lines)


def wizard(directory=None):
    """An optional terminal questionnaire. Unknown answers are never guessed."""
    if directory is None:
        print("Available: Arm joint tuning. Planned, not executable: grasp; insertion.")
        choice = input("Recipe [arm_joint_response@1]: ").strip() or ARM_ID
        asset = input("Robot USD path: ").strip()
        goal = input("What task are you preparing the robot for? ").strip()
        evidence = input("Existing evidence descriptor JSON [blank if none yet]: ").strip() or None
        directory = input("New session folder [runs/guided-<id>]: ").strip() or f"runs/guided-{uuid.uuid4().hex[:8]}"
        state = start(asset=asset, goal=goal, directory=directory, recipe=choice, evidence=evidence)
    else:
        state = review(directory)
    print(f"Session: {state['session_id']} — {state['state'].replace('_', ' ')}")
    print(
        "An agent can help prepare the complex answers. Paste JSON or a JSON-file path; enter ? to leave an answer unknown."
    )
    # Each question is offered at most once in this invocation; unknown does not
    # trap a user in an endless loop or become a confirmation.
    offered = set()
    answer_keys = {"recipe", "controller", "environment", "tool", "collection", "evidence", "joint_bindings"}
    while True:
        state = advance(directory) if state["recipe_id"] else review(directory)
        remaining = [
            item
            for item in state.get("questions", [])
            if item["key"] not in offered and (item["key"] in answer_keys or item["key"].startswith("confirm."))
        ]
        if not remaining:
            break
        item = remaining[0]
        key = item["key"]
        offered.add(key)
        print(item["why"])
        if key.startswith("confirm."):
            print("Review current proposals and answers before confirming:")
            current = review(directory)
            print(json.dumps({"inputs": current["inputs"], "proposals": current["proposals"]}, indent=2))
            answer = input(item["question"] + " [yes / no / ?]: ").strip().lower()
            if answer != "yes":
                continue
            by = input("Who verified this? ").strip()
            source = input("Source of the confirmation: ").strip()
            state = provide(directory, {"confirm": [key.split(".", 1)[1]]}, source=source, confirmed_by=by)
            continue
        if item.get("example") is not None:
            print("Example:", json.dumps(item["example"], indent=2))
        answer = input(item["question"] + "\n> ").strip()
        if not answer or answer == "?":
            continue
        if key == "recipe":
            value = answer
        else:
            value = json.loads(answer) if answer.startswith(("{", "[")) else read(Path(answer).expanduser())
        source = input("Where did this information come from? ").strip()
        state = provide(directory, {key: value}, source=source)
    if (
        state["state"] in {"ready_to_fit", "ready_to_resume"}
        and input("Ready. Run calibration in simulation now? [yes / no]: ").strip().lower() == "yes"
    ):
        state = advance(directory, execute=True)
    print(describe(state))
    print(f"Resume: newton-calibration guide wizard --session {json.dumps(str(directory))}")
    print("Use guide advance to inspect/prepare collection; add --execute only when ready to fit.")
    return state


def run(args):
    action = args.guide_action
    if action is None:
        wizard()
        return
    if action == "recipes":
        items = list_recipes()
        print(
            json.dumps(items, indent=2)
            if args.json
            else "\n".join(f"{r['name']} [{r['status']}] — {r['id']}\n  {r['description']}" for r in items)
        )
        return
    if action == "start":
        state = start(
            asset=args.asset, goal=args.goal, directory=args.session, recipe=args.recipe, evidence=args.evidence
        )
    elif action == "review":
        state = review(args.session)
    elif action == "status":
        state = status(args.session)
    elif action == "provide":
        state = provide(args.session, read(args.answers), source=args.source, confirmed_by=args.confirmed_by)
    elif action == "wizard":
        wizard(args.session)
        return
    else:
        preview = _factory(args.preview_factory) if args.preview_factory else None
        probe = (
            _factory(args.design_probe_factory) if args.design_probe_factory else getattr(preview, "design_probe", None)
        )
        state = advance(args.session, execute=args.execute, preview=preview, design_probe=probe)
    print(json.dumps(state, indent=2) if args.json else describe(state))
