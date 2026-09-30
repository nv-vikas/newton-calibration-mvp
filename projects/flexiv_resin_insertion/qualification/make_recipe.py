"""Create an explicit recipe from an already prepared, unused reset bank."""

import argparse
from dataclasses import asdict
from pathlib import Path

from newton_calibration.core.io import atomic_write_json
from newton_calibration.qualification import Gates, Recipe, Settings

from .worker import input_files, read, selected_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep previous recipe revisions")
    binding = read(args.binding)
    _, job = selected_job(binding)
    available = {str(row["source_heldout_index"]) for row in job["starts"]}
    if not set(args.development + args.validation) <= available:
        raise ValueError("Prepare all requested start-bank indices first")
    # This family keeps gap (detection) distinct from the contact surface margin.
    # Candidates are hypotheses, not claims that these values represent real materials.
    candidates = (
        Settings(margin_m=5e-6),
        Settings(margin_m=0),
        Settings(margin_m=0, gap_m=10e-6),
        Settings(dt_s=1 / 1920, margin_m=0, gap_m=10e-6, iterations=200, tolerance=1e-6),
    )
    recipe = Recipe(
        name=f"{binding['hole']} precision insertion qualification",
        inputs=input_files(args.binding, binding),
        development_trials=tuple(args.development),
        validation_trials=tuple(args.validation),
        candidates=candidates,
        max_candidates=len(candidates),
        timeout_s=1800,
        gates=Gates(),
        record_video=True,
    )
    atomic_write_json(args.output, asdict(recipe))


if __name__ == "__main__":
    main()
