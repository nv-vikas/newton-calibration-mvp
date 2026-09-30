"""Data-only contracts for simulation qualification, separate from real-data fitting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < 0 if zero else value <= 0:
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class Settings:
    """Only numerical settings; geometry, materials, actions and gains are not tunable here."""

    dt_s: float = 1 / 960
    margin_m: float = 50e-6
    gap_m: float = 50e-6
    iterations: int = 100
    tolerance: float = 1e-6

    def __post_init__(self):
        for name in ("dt_s", "tolerance"):
            positive(getattr(self, name), name)
        for name in ("margin_m", "gap_m"):
            positive(getattr(self, name), name, zero=True)
        if type(self.iterations) is not int or not 1 <= self.iterations <= 1000:
            raise ValueError("iterations must be an integer in [1, 1000]")

    def validate_rate(self, hz):
        decimation = 1 / (self.dt_s * hz)
        if not 1 <= decimation <= 4096 or abs(decimation - round(decimation)) > 1e-7:
            raise ValueError("dt must preserve the locked policy rate with an integer decimation <= 4096")

    def refined(self, *, tolerance_floor=0.0):
        positive(tolerance_floor, "tolerance_floor", zero=True)
        return Settings(
            self.dt_s / 2,
            self.margin_m,
            self.gap_m,
            min(self.iterations * 2, 1000),
            max(self.tolerance / 10, tolerance_floor),
        )


@dataclass(frozen=True)
class Gates:
    insertion_depth_m: float = 0.035
    maximum_depth_m: float = 0.041
    hold_s: float = 0.25
    max_grasp_displacement_m: float = 0.003
    max_wall_penetration_m: float = 5e-6
    repeatability_m: float = 5e-6
    convergence_m: float = 5e-6
    max_geometry_error_m: float = 2e-6
    rim_max_depth_m: float = 0.003
    minimum_policy_success_fraction: float = 0.75

    def __post_init__(self):
        for name, value in asdict(self).items():
            positive(value, name)
        if self.maximum_depth_m <= self.insertion_depth_m:
            raise ValueError("maximum depth must exceed insertion depth")
        if self.minimum_policy_success_fraction > 1:
            raise ValueError("policy success fraction must be <= 1")


@dataclass(frozen=True)
class Recipe:
    name: str
    inputs: dict[str, str]
    development_trials: tuple[str, ...]
    validation_trials: tuple[str, ...]
    baseline: Settings = field(default_factory=Settings)
    candidates: tuple[Settings, ...] = ()
    gates: Gates = field(default_factory=Gates)
    policy_hz: float = 30.0
    max_candidates: int = 8
    timeout_s: float = 900.0
    record_video: bool = True
    optimizer: dict[str, Any] | None = None
    schema: str = "newton.qualification/precision-insertion-v1"

    def __post_init__(self):
        if self.schema != "newton.qualification/precision-insertion-v1" or not self.name.strip():
            raise ValueError("Unknown recipe schema or empty name")
        if not {"asset", "policy", "controller", "scene", "trial_bank"} <= self.inputs.keys():
            raise ValueError("Pin asset, policy, controller, scene and trial_bank inputs (plus their dependencies)")
        if any(not isinstance(p, str) or not p for p in self.inputs.values()):
            raise ValueError("Inputs must be explicit file paths")
        positive(self.policy_hz, "policy_hz")
        positive(self.timeout_s, "timeout_s")
        if (
            type(self.record_video) is not bool
            or type(self.max_candidates) is not int
            or not 1 <= self.max_candidates <= 64
        ):
            raise ValueError("Invalid video flag or candidate budget (1..64)")
        for ids in (self.development_trials, self.validation_trials):
            if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("Each split requires nonempty unique trial IDs")
        if set(self.development_trials) & set(self.validation_trials):
            raise ValueError("Development and validation trials must be disjoint")
        if len(self.candidates) > self.max_candidates:
            raise ValueError("Candidate list exceeds the locked budget")
        for settings in (self.baseline, *self.candidates):
            settings.validate_rate(self.policy_hz)
            settings.refined().validate_rate(self.policy_hz)
        if self.optimizer and self.candidates:
            raise ValueError("Use a declared sweep OR an optimizer, not both")

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["baseline"] = Settings(**data.get("baseline", {}))
        data["candidates"] = tuple(Settings(**c) for c in data.get("candidates", []))
        data["gates"] = Gates(**data.get("gates", {}))
        for key in ("development_trials", "validation_trials"):
            data[key] = tuple(data[key])
        return cls(**data)


class Backend(Protocol):
    """Trusted worker, never an agent-generated import embedded in evidence/recipes.

    describe includes runtime/source identity, supported settings and trial content
    fingerprints. execute returns measurements, not a self-declared pass/fail.
    No training or hardware operation exists on this interface.
    """

    def describe(self) -> dict[str, Any]: ...

    def execute(self, request: dict[str, Any], output_dir: str) -> dict[str, Any]: ...
