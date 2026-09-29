"""python -m newton_calibration.reporting --recipe ... --bundle ... --output ..."""

import argparse

from .report import build_report, render_report


def main():
    parser = argparse.ArgumentParser(
        description="Render a customer report from recipe requirements and saved run records"
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--strict", action="store_true", help="Render gaps, then return exit code 2 if required facts are missing"
    )
    args = parser.parse_args()
    model = build_report(args.recipe, args.bundle)
    path = render_report(model, args.output)
    print(f"Report: {path}; required facts {'complete' if model['report_complete'] else 'incomplete'}")
    if args.strict and not model["report_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
