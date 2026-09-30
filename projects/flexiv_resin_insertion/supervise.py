"""Bounded detached workflow. Explicit result files, not process exit alone, gate stages."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_status(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def command(script, prepared, output, mode, num_envs, hours):
    return [
        "/workspace/isaaclab/isaaclab.sh",
        "-p",
        str(script),
        "--mode",
        mode,
        "--prepared",
        str(prepared),
        "--output",
        str(output),
        "--num_envs",
        str(num_envs),
        "--max_wall_hours",
        str(hours),
        "--resume",
        "--headless",
    ]


def qualification_is_current(output, started_at):
    qualification = read_status(Path(output) / "qualification.json")
    status = read_status(Path(output) / "status.json")
    return (
        qualification.get("passed") is True
        and status.get("state") == "qualified"
        and status.get("updated_unix", 0) >= started_at
        and bool(qualification.get("experiment_fingerprint"))
        and qualification["experiment_fingerprint"] == status.get("experiment_fingerprint")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--hours", type=float, default=48.0)
    args = parser.parse_args()
    if args.hours <= 0 or args.hours > 48 or args.num_envs < 1:
        parser.error("Use a positive budget no greater than 48 hours and at least one environment")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    script = Path(__file__).with_name("run.py")
    deadline = time.monotonic() + args.hours * 3600
    child = None
    stop = False

    def stopping(_signum, _frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)

    def run(mode, num_envs, attempt=0):
        nonlocal child, stop
        hours = (deadline - time.monotonic()) / 3600
        if hours <= 0 or stop:
            return False
        with (args.output / f"{mode}_{attempt}.log").open("a") as log:
            child = subprocess.Popen(
                command(script, args.prepared, args.output, mode, num_envs, hours),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(json.dumps({"stage": mode, "pid": child.pid, "attempt": attempt}), flush=True)
            while child.poll() is None:
                if time.monotonic() >= deadline or stop:
                    stop = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    return False
                time.sleep(5)
        return child.returncode == 0

    # Requalify in this exact runtime, never trust an old declaration alone.
    qualification_started_at = time.time()
    run("qualify", 1)
    if not qualification_is_current(args.output, qualification_started_at):
        print("BLOCKED: physical qualification did not pass; no training launched.", flush=True)
        return 2
    for attempt in range(3):
        run("train", args.num_envs, attempt)
        state = read_status(args.output / "status.json").get("state")
        if state == "training_complete_needs_evaluation":
            run("evaluate", min(args.num_envs, 4))
            return 0 if read_status(args.output / "status.json").get("state") == "evaluated" else 3
        if state in {"budget_exhausted", "stopped_checkpointed"} or stop:
            return 0
        if not (args.output / "latest_checkpoint.json").exists():
            print("BLOCKED: training failed before a checkpoint; manual diagnosis required.", flush=True)
            return 4
    return 5


if __name__ == "__main__":
    sys.exit(main())
