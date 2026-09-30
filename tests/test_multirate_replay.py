"""Timestamp replay tests, not real-controller equivalence or Newton physics proof."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newton_calibration.adapters.evidence import TabularJointEvidence
from newton_calibration.core import JointBinding, LongFormSchema, SignalBinding, bind_evidence_files


def evidence_at(root: Path, times):
    rows = []
    times = np.asarray(times, dtype=float)
    for split, offset in (("train", 0), ("heldout", 0.2)):
        for signal, values in (
            ("command_q", np.arange(len(times)) + offset),
            ("actual_q", times + offset),
            ("actual_dq", np.ones(len(times))),
        ):
            rows.extend({"time": t, "joint": "axis", "signal": signal, "value": v} for t, v in zip(times, values))
        pd.DataFrame(rows).to_csv(root / f"{split}.csv", index=False)
        rows.clear()
    spec = bind_evidence_files(
        root=root,
        revision="synthetic-multirate@1",
        episodes=[
            {"name": split, "path": f"{split}.csv", "split": split, "trial_id": split} for split in ("train", "heldout")
        ],
        schema=LongFormSchema(
            time_column="time", time_unit="s", joint_column="joint", signal_column="signal", field_column=None
        ),
        joint_bindings=(JointBinding("axis", "axis", "rad", "rad", transform_confirmed=True),),
        signal_bindings=tuple(SignalBinding(key, key) for key in ("command_q", "actual_q", "actual_dq")),
    )
    return TabularJointEvidence(spec)


def test_30hz_commands_are_held_for_32_ticks_at_960hz(tmp_path):
    evidence = evidence_at(tmp_path, np.arange(7) / 30)
    episode = evidence.load_episode("train", dt=1 / 960)
    np.testing.assert_allclose(episode.command_q[:160, 0], np.repeat(np.arange(5), 32))
    assert evidence.replay_timing(1 / 960)["supported"]
    # Interpolation supplies comparison values; it is not counted as real samples.
    assert evidence.replay_timing(1 / 960)["streams"][0]["command_samples"] == 7


@pytest.mark.parametrize(
    "times,dt",
    [
        (np.arange(7) / 30, 1 / 1000),  # ratio is not an integer
        ([0, 0.0334, 0.0682, 0.1021, 0.136, 0.17], 1 / 960),  # real timestamp jitter
        ([0, 0.0334, 0.0682, 0.3021, 0.336, 0.37], 1 / 960),  # hold across a long gap
    ],
)
def test_timestamp_schedule_not_a_fixed_repeat_count(tmp_path, times, dt):
    evidence = evidence_at(tmp_path, times)
    episode = evidence.load_episode("train", dt=dt)
    source_times = np.asarray(times)
    expected = [max(i for i, source_t in enumerate(source_times) if source_t <= t) for t in episode.time_s]
    np.testing.assert_array_equal(episode.command_q[:, 0], expected)
    for index in np.flatnonzero(np.diff(episode.command_q[:, 0])) + 1:
        source_index = int(episode.command_q[index, 0])
        delay = episode.time_s[index] - source_times[source_index]
        assert -1e-12 <= delay < dt + 1e-12
    assert evidence.replay_timing(dt)["supported"]


def test_simulation_cannot_silently_skip_faster_commands(tmp_path):
    evidence = evidence_at(tmp_path, [0, 0.0005, 0.0015, 0.0025])
    timing = evidence.replay_timing(0.001)
    assert not timing["supported"] and timing["blockers"]
    with pytest.raises(ValueError, match="faster than the physics step"):
        evidence.load_episode("train", dt=0.001)


def test_nominally_equal_rate_allows_only_floating_point_roundoff(tmp_path):
    evidence = evidence_at(tmp_path, 1_000_000 + np.arange(8) / 30)
    assert evidence.replay_timing(1 / 30)["supported"]


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_timing_rejected(tmp_path, dt):
    evidence = evidence_at(tmp_path, np.arange(6) / 30)
    with pytest.raises(ValueError, match="finite"):
        evidence.replay_timing(dt)


def test_source_commands_unchanged_by_resampling(tmp_path):
    evidence = evidence_at(tmp_path, np.arange(7) / 30)
    before = evidence.spec.fingerprint
    evidence.load_episode("train", dt=1 / 960)
    evidence.load_episode("train", dt=1 / 1000)
    evidence.verify_files()
    assert evidence.spec.fingerprint == before
