"""Offline assistance must never confirm setup facts or overwrite inputs."""

import json

import numpy as np
import pytest

from newton_calibration.guided.preflight import inspect_simulation_profile, replay_slew_limiter


def profile(tmp_path):
    path = tmp_path / "simulation.json"
    data = {
        "joint_map": {"a": "shoulder"},
        "base_stiffness_by_joint": {"a": 12.0},
        "base_damping_by_joint": {"a": 0.7},
        "base_effort_limit_by_joint": {"a": 2.0},
        "profile_confirmed": True,  # even an existing flag cannot attest this run
    }
    path.write_text(json.dumps(data))
    return path, data


def test_explicit_simulation_profile_is_a_proposal_only(tmp_path):
    path, data = profile(tmp_path)
    before = path.read_bytes()
    result = inspect_simulation_profile(path, data["joint_map"])
    assert result["complete"] and not result["hardware_confirmed"] and not result["runtime_validated"]
    assert "profile_confirmed" not in result["environment_patch"]
    assert result["source"]["sha256"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [None, {}, {"b": 1.0}, {"a": -1}, {"a": float("nan")}, {"a": True}])
def test_bad_profile_never_uses_scalar_defaults(tmp_path, bad):
    path, data = profile(tmp_path)
    data["base_damping_by_joint"] = bad
    data["base_damping"] = 10.0
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        inspect_simulation_profile(path, data["joint_map"])


def test_profile_rejects_map_mismatch(tmp_path):
    path, _ = profile(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        inspect_simulation_profile(path, {"a": "elbow"})


def test_limiter_handles_initial_state_jitter_and_reversal():
    targets = np.array([[0.2], [0.2], [-0.2], [0.0]])
    before = targets.copy()
    result = replay_slew_limiter(
        [0, 0.1, 0.13, 0.1301], targets, velocity_limit=0.4, initial_target=[0], initial_time=-0.1
    )
    np.testing.assert_allclose(result[:, 0], [0.04, 0.08, 0.068, 0.0676])
    np.testing.assert_array_equal(targets, before)


def test_first_target_passthrough_is_a_different_explicit_assumption():
    result = replay_slew_limiter([0, 0.1], [[0.2], [0.3]], velocity_limit=0.4)
    np.testing.assert_allclose(result[:, 0], [0.2, 0.24])
    initialized = replay_slew_limiter(
        [0, 0.1], [[0.2], [0.3]], velocity_limit=0.4, initial_target=[0], initial_time=-0.1
    )
    assert not np.allclose(result, initialized)


@pytest.mark.parametrize(
    "times,targets,kwargs",
    [
        ([], [], {}),
        ([0, 0], [[0], [1]], {}),
        ([0, 1], [[0]], {}),
        ([0], [[float("nan")]], {}),
        ([0], [[0]], {"initial_target": [0]}),
        ([0], [[0]], {"initial_target": [0], "initial_time": 1}),
        ([0], [[0]], {"minimum_dt": 0}),
        ([0], [[0]], {"minimum_dt": float("inf")}),
    ],
)
def test_limiter_rejects_ambiguous_or_invalid_inputs(times, targets, kwargs):
    with pytest.raises(ValueError):
        replay_slew_limiter(times, targets, velocity_limit=0.4, **kwargs)


def test_limiter_matches_independent_scalar_reference():
    rng = np.random.default_rng(42)
    times = np.cumsum(rng.uniform(0.0001, 0.15, size=100))
    targets = rng.uniform(-1, 1, size=(100, 7))
    actual = replay_slew_limiter(times, targets, velocity_limit=0.4)
    expected = [targets[0].tolist()]
    for i in range(1, len(times)):
        step = 0.4 * max(times[i] - times[i - 1], 0.001)
        expected.append([max(q - step, min(q + step, p)) for p, q in zip(targets[i], expected[-1])])
    np.testing.assert_allclose(actual, expected, atol=1e-15)
