"""Launch the isolated reference qualification in an existing pinned Docker runtime."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from newton_calibration.core.io import atomic_write_json
from newton_calibration.qualification import QualificationJob, Recipe

from .factory import create_backend
from .worker import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact frozen checkpoint, not latest")
    parser.add_argument("--source", type=Path, default=Path("/work/flexiv_resin_insertion"))
    parser.add_argument("--hole", default="chamfer_2505")
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "binding.json"
    if binding_path.exists():
        raise FileExistsError("Use the qualify CLI to resume; bootstrap never overwrites an experiment")
    tools = Path(__file__).resolve().parents[1] / "tools"
    subprocess.run(
        [
            sys.executable,
            str(tools / "prepare_other_holes.py"),
            "--prepared",
            str(args.prepared),
            "--output",
            str(output / "starts.json"),
            "--indices",
            "0",
            "1",
            "3",
            "4",
            "--holes",
            args.hole,
        ],
        check=True,
    )
    atomic_write_json(
        binding_path,
        {
            "source": str(args.source),
            "tools": str(tools),
            "prepared": str(args.prepared),
            "checkpoint": str(args.checkpoint),
            "ppo_config": str(args.training_run / "ppo_config.json"),
            "spec": str(output / "starts.json"),
            "hole": args.hole,
            "device": "cuda:0",
            "scope": "source starts 0,1 for diagnostics; 3,4 reserved for frozen-policy evaluation; no retraining",
        },
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "projects.flexiv_resin_insertion.qualification.make_recipe",
            "--binding",
            str(binding_path),
            "--output",
            str(output / "recipe.json"),
            "--development",
            "0",
            "1",
            "--validation",
            "3",
            "4",
        ],
        check=True,
    )
    os.environ["NEWTON_QUALIFICATION_BINDING"] = str(binding_path)
    backend = create_backend()
    job = QualificationJob.create(Recipe.from_dict(read(output / "recipe.json")), backend, output / "job")
    result = job.advance(expected_stage="geometry") if args.geometry_only else job.run()
    print(result, flush=True)
    if result["state"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
