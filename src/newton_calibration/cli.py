from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from newton_calibration.adapters.evidence import fetch_anchor_lab_so101
from newton_calibration.adapters.surface import SO101EnvCfg
from newton_calibration.collection import CalibrationRequest, MotionSpec
from newton_calibration.controllers import (
    ControllerProfile,
    controller_report,
    discover_controller,
    prompt_controller,
    resolve_controller,
)
from newton_calibration.core.models import EnvironmentSpec, jsonable
from newton_calibration.isaaclab import tuning
from newton_calibration.optimizers import list_optimizers


def main() -> None:
    parser = argparse.ArgumentParser(prog="newton-calibration", description="Newton calibration MVP1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    from newton_calibration.qualification.cli import add_parser as add_qualification_parser

    add_qualification_parser(subparsers)
    fetch_parser = subparsers.add_parser("fetch", help="download SO-101 Anchor-Lab evidence and USD")
    fetch_parser.add_argument("--output", default="data/anchor-lab")
    fetch_parser.add_argument("--revision", default="647edd5787cd764cdc041103ad282dc59214d919")
    subparsers.add_parser("optimizers", help="list installed optimizer plug-ins and versions")
    controller_parser = subparsers.add_parser(
        "controller", help="Discover a controller and optionally confirm a data-only profile; never runs calibration"
    )
    controller_parser.add_argument(
        "--env-factory", help="Trusted local module:factory returning an Isaac Lab env config (not a USD)"
    )
    controller_parser.add_argument("--profile", help="Existing controller-profile JSON to review")
    controller_parser.add_argument(
        "--action-name", help="Select an action when the environment has multiple action terms"
    )
    controller_parser.add_argument("--asset-name", default="robot")
    controller_parser.add_argument("--name", default="controller-profile")
    controller_parser.add_argument(
        "--output", required=True, help="New controller-profile JSON; existing files are never overwritten"
    )
    controller_parser.add_argument(
        "--interactive", action="store_true", help="Ask for missing settings and explicit confirmation in a terminal"
    )
    assist_parser = subparsers.add_parser(
        "assist", help="asset-only analyze → collection plan, with video enabled by default"
    )
    assist_parser.add_argument(
        "--config", required=True, help="JSON containing environment and collection (MotionSpec) objects"
    )
    assist_parser.add_argument(
        "--preview-factory", help="Trusted local module:factory returning a bound Isaac Lab/Newton preview adapter"
    )
    assist_parser.add_argument(
        "--design-probe-factory",
        help="Trusted local module:factory returning a Newton sensitivity probe bound to the same scene",
    )
    assist_parser.add_argument("--workdir", default="runs")
    assist_parser.add_argument("--no-preview", action="store_true", help="Explicitly skip the default preview request")
    assist_parser.add_argument("--controller-profile", help="Data-only controller-profile JSON")
    assist_parser.add_argument(
        "--controller-env-factory", help="Trusted local module:factory returning the active Isaac Lab config"
    )
    assist_parser.add_argument(
        "--controller-action", help="Action term to inspect when more than one controls the robot"
    )

    inspect_parser = subparsers.add_parser("inspect", help="execute analyze + plan without starting physics")
    _add_job_arguments(inspect_parser, include_fit=False)

    run_parser = subparsers.add_parser("run", help="execute analyze → plan → fit → validate → write")
    _add_job_arguments(run_parser, include_fit=True)

    args = parser.parse_args()
    if args.command == "qualify":
        from newton_calibration.qualification.cli import execute

        execute(args)
        return
    if args.command == "controller":
        from dataclasses import replace

        if args.interactive and not sys.stdin.isatty():
            parser.error("--interactive requires a terminal; omit it to return machine-readable questions")
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            parser.error("Controller output already exists; choose a new revision path")
        profile = ControllerProfile.load(args.profile) if args.profile else None
        if args.env_factory:
            cfg = _trusted_factory(args.env_factory)
            if profile is not None:
                # Same conflict checks as analyze: stale confirmations cannot
                # silently relabel a freshly discovered controller.
                resolved, _ = resolve_controller(
                    EnvironmentSpec(adapter="isaaclab_newton", asset_path="discovery-only"),
                    profile=profile,
                    isaaclab_env=cfg,
                    action_name=args.action_name,
                    asset_name=args.asset_name,
                )
                profile = ControllerProfile.from_dict(resolved.controller_profile)
            else:
                profile = replace(
                    discover_controller(cfg, action_name=args.action_name, asset_name=args.asset_name), name=args.name
                )
        profile = profile or ControllerProfile(name=args.name)
        if args.interactive:
            profile = prompt_controller(profile, show=lambda value: print(value, file=sys.stderr))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(profile.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")
        report = controller_report(profile)
        print(json.dumps({"profile_path": str(output), **report}, indent=2))
        if not report["fit_ready"]:
            raise SystemExit(2)
        return
    if args.command == "fetch":
        print(json.dumps(fetch_anchor_lab_so101(args.output, args.revision), indent=2))
        return
    if args.command == "optimizers":
        print(json.dumps(list_optimizers(), indent=2, sort_keys=True))
        return
    if args.command == "assist":
        import importlib

        config = json.loads(Path(args.config).read_text())
        preview = None
        if args.preview_factory:
            module, separator, name = args.preview_factory.partition(":")
            if not separator:
                parser.error("--preview-factory must be a trusted local module:factory")
            preview = getattr(importlib.import_module(module), name)()
        design_probe = getattr(preview, "design_probe", None)
        if args.design_probe_factory:
            module, separator, name = args.design_probe_factory.partition(":")
            if not separator:
                parser.error("--design-probe-factory must be a trusted local module:factory")
            design_probe = getattr(importlib.import_module(module), name)()
        result = tuning.assist(
            env=EnvironmentSpec(**config["environment"]),
            request=CalibrationRequest(**config.get("request", {})),
            collection=MotionSpec(**config["collection"]) if config.get("collection") else None,
            preview=preview,
            design_probe=design_probe,
            video=not args.no_preview,
            controller_profile=ControllerProfile.load(args.controller_profile)
            if args.controller_profile
            else config.get("controller_profile"),
            isaaclab_env=_trusted_factory(args.controller_env_factory) if args.controller_env_factory else None,
            controller_action=args.controller_action,
            workdir=args.workdir,
        )
        print(json.dumps(jsonable(result), indent=2))
        if (
            result.status
            in {
                "preview_failed",
                "preview_pending",
                "needs_scene_setup",
                "generation_failed",
                "evidence_action_required",
                "design_failed",
                "controller_action_required",
            }
            or result.design.get("adaptive_search", {}).get("status") == "needs_dynamics_probe"
        ):
            raise SystemExit(2)  # files may exist, but requested design/preview is not complete
        return
    env = SO101EnvCfg(
        usd_path=str(Path(args.asset).expanduser().resolve()),
        runtime=args.runtime,
        device=args.device,
        residual_model_path=args.residual_model,
    )
    analysis = tuning.analyze(
        env=env,
        evidence=args.evidence,
        evidence_revision=args.revision,
        controller_profile=ControllerProfile.load(args.controller_profile) if args.controller_profile else None,
        workdir=args.workdir,
    )
    calibration_plan = tuning.plan(
        analysis,
        optimizer=args.optimizer,
        optimizer_options=args.optimizer_options_json,
    )
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "analysis": jsonable(analysis),
                    "plan": jsonable(calibration_plan),
                },
                indent=2,
            )
        )
        return
    fit_run = tuning.fit(
        calibration_plan,
        generations=args.generations,
        population=args.population,
        resume=not args.no_resume,
    )
    validation = tuning.validate(fit_run)
    package = tuning.write(validation, output=args.output)
    print(json.dumps(jsonable(package), indent=2))


def _add_job_arguments(parser: argparse.ArgumentParser, *, include_fit: bool) -> None:
    parser.add_argument(
        "--controller-profile", help="Data-only controller-profile JSON; unsupported controllers fail closed"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--workdir", default="runs")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--runtime", choices=["isaaclab_newton", "analytic"], default="isaaclab_newton")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--residual-model")
    parser.add_argument(
        "--optimizer",
        help="registered optimizer name; defaults to the recipe optimizer",
    )
    parser.add_argument(
        "--optimizer-options-json",
        type=_json_object,
        metavar="JSON",
        help="optimizer-specific JSON object locked into the calibration plan",
    )
    if include_fit:
        parser.add_argument("--output", default="packages/so101")
        parser.add_argument("--generations", type=int)
        parser.add_argument("--population", type=int)
        parser.add_argument("--no-resume", action="store_true")


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid optimizer JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("optimizer options must be a JSON object")
    return parsed


def _trusted_factory(value: str):
    """Explicit executable input only; never import a path from profile data."""
    import importlib

    module, separator, name = value.partition(":")
    if not separator or not module or not name:
        raise ValueError("Controller factory must be a trusted local module:factory")
    return getattr(importlib.import_module(module), name)()


if __name__ == "__main__":
    main()
