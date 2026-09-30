"""CLI and machine-readable agent entry point; executable backend selection is explicit."""

import importlib
import json
from pathlib import Path

from .contracts import Recipe
from .runner import STAGES, QualificationJob


def add_parser(subparsers):
    parser = subparsers.add_parser("qualify", help="Four-stage simulation qualification; never trains or runs hardware")
    parser.add_argument("action", choices=["create", "status", "advance", "run"])
    parser.add_argument("--job", required=True, help="New job directory for create; existing directory otherwise")
    parser.add_argument("--recipe", help="Data-only JSON recipe; required only for create")
    parser.add_argument("--backend-factory", required=True, help="Explicit trusted Python module:factory")
    parser.add_argument("--expected-stage", choices=STAGES, help="Optional guard for an agent's advance call")


def execute(args):
    module, separator, factory = args.backend_factory.partition(":")
    if not separator or not module or not factory:
        raise ValueError("Use a trusted module:factory, never executable paths from evidence")
    backend = getattr(importlib.import_module(module), factory)()
    if args.action == "create":
        if not args.recipe:
            raise ValueError("create requires --recipe")
        job = QualificationJob.create(Recipe.from_dict(json.loads(Path(args.recipe).read_text())), backend, args.job)
        result = job.status()
    else:
        if args.recipe:
            raise ValueError("Existing jobs use their locked recipe; create a new revision to change it")
        job = QualificationJob(args.job, backend)
        if args.action == "status":
            result = job.status()
        elif args.action == "advance":
            result = job.advance(expected_stage=args.expected_stage)
        else:
            result = job.run()
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["state"] in {"blocked", "execution_failed"}:
        raise SystemExit(2)
