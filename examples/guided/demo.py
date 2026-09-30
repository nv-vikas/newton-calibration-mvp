"""Synthetic guided journey: collection proposal -> evidence -> five calls.

No Newton/Isaac Lab runtime, hardware, or real measurements. Fixture motions must
never be used on a robot. Existing destinations are refused, not overwritten.
"""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from newton_calibration import guided
from newton_calibration.core.io import write_json


def create_fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    asset = root / "synthetic-arm.usda"
    asset.write_text(
        '#usda 1.0\ndef Xform "Robot" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {\n'
        'def PhysicsRevoluteJoint "shoulder" {}\ndef PhysicsRevoluteJoint "elbow" {}\n}\n'
    )
    evidence_dir = root / "synthetic-evidence"
    evidence_dir.mkdir()
    for name, phase in (("training", 0.0), ("heldout", 0.43)):
        with (evidence_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "joint", "signal", "value"])
            t = np.arange(0, 2.0, 0.02)
            for joint in ("a", "b"):
                q = 0.4 * np.sin(2 * np.pi * 0.75 * t + phase)
                for signal, values in (
                    ("command_q", q),
                    ("actual_q", q * 0.85),
                    ("actual_dq", np.gradient(q * 0.85, 0.02)),
                ):
                    writer.writerows((float(time), joint, signal, float(value)) for time, value in zip(t, values))
    evidence = {
        "root": str(evidence_dir),
        "schema": {
            "time_column": "time",
            "time_unit": "s",
            "value_column": "value",
            "joint_column": "joint",
            "signal_column": "signal",
            "field_column": None,
        },
        "episodes": [
            {"name": name, "path": f"{name}.csv", "split": split, "trial_id": f"synthetic-{name}"}
            for name, split in (("training", "train"), ("heldout", "heldout"))
        ],
        "signal_bindings": [
            {"source_signal": name, "canonical_signal": name} for name in ("command_q", "actual_q", "actual_dq")
        ],
    }
    answers = {
        "environment": {
            "adapter": "analytic",
            "device": "cpu",
            "robot_id": "SYNTHETIC test arm",
            "dt": 0.02,
            "joint_map": {"a": "shoulder", "b": "elbow"},
            "joint_groups": {"arm": ["a", "b"]},
            "base_stiffness_by_joint": {"a": 20.0, "b": 20.0},
            "base_damping_by_joint": {"a": 1.0, "b": 1.0},
            "base_effort_limit_by_joint": {"a": 20.0, "b": 20.0},
            "analytic_inertia_by_joint": {"a": 1.0, "b": 1.0},
        },
        "controller": {
            "simulation_mode": "joint_position_pd",
            "real_mode": "joint_position",
            "command_semantics": "absolute_joint_position",
            "command_unit": "rad",
            "command_rate_hz": 50,
            "source": "Synthetic fixture, no real controller",
            "filters": "none",
            "gravity_compensation": "test only",
        },
        "tool": {"kind": "none", "matches_asset": True, "source": "Synthetic fixture"},
        "joint_bindings": [
            {
                "source_joint": source,
                "usd_joint": target,
                "source_unit": "rad",
                "usd_unit": "rad",
                "sign": 1,
                "scale": 1.0,
                "offset": 0.0,
            }
            for source, target in (("a", "shoulder"), ("b", "elbow"))
        ],
        "evidence": None,
        "collection": {
            "joint_names": ["shoulder", "elbow"],
            "center_rad": [0.0, 0.0],
            "lower_rad": [-2.0, -2.0],
            "upper_rad": [2.0, 2.0],
            "amplitude_rad": [0.3, 0.3],
            "max_velocity_rad_s": [0.4, 0.4],
            "max_acceleration_rad_s2": [0.6, 0.6],
            "source": "Synthetic limits, NEVER USE ON HARDWARE",
            "scene_id": "synthetic-fixture",
            "command_rate_hz": 50,
            "duration_s": 12.0,
        },
        "fit_budget": {"generations": 2, "population": 4},
        "confirm": ["mapping", "controller", "tool"],
    }
    return asset, answers, evidence


def demo(destination):
    root = Path(destination).expanduser().resolve()
    asset, answers, evidence = create_fixture(root)
    print("SYNTHETIC software-flow demo. No robot or Newton GPU run.")
    session_dir = root / "session"
    session = guided.start(asset=asset, goal="Prepare an arm for later insertion; fixture only", directory=session_dir)
    print("1.", session["state"], "— arm joint tuning is available; grasp/insertion are planned")
    guided.provide(session_dir, {"recipe": "arm_joint_response@1"}, source="Demo choice")
    session = guided.run(session_dir)
    print("2.", session["state"], "— analysis completed; setup questions remain")
    write_json(root / "setup-answers.SYNTHETIC.json", answers)
    guided.provide(
        session_dir,
        answers,
        source="Synthetic fixture definitions",
        confirmed_by="Synthetic test fixture (not a person)",
    )
    session = guided.run(session_dir)
    print("3.", session["state"], "—", len(session["collection"]["episodes"]), "motion proposals; no GPU preview")
    attached = {"evidence": deepcopy(evidence), "confirm": ["mapping"]}
    write_json(root / "attach-evidence.SYNTHETIC.json", attached)
    guided.provide(
        session_dir,
        attached,
        source="Generated synthetic CSVs, never hardware",
        confirmed_by="Synthetic test fixture (not a person)",
    )
    session = guided.run(session_dir)
    print("4.", session["state"], "— independent synthetic train/heldout records")
    session = guided.run(session_dir, execute=True)
    print(
        "5.",
        session["state"],
        "— five actual API calls on analytic backend; activation:",
        session["activation_allowed"],
    )
    print("Customer report:", session["customer_report"])
    print("Session:", session_dir / "session.json")
    return session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/guided-demo")
    demo(parser.parse_args().output)
