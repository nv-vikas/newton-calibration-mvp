"""Regression checks for sample timing, not validation of Newton physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from newton_calibration.adapters.runtime.analytic import AnalyticPDReplayRuntime
from newton_calibration.adapters.runtime.newton import IsaacLabNewtonRuntime
from newton_calibration.core.models import EnvironmentSpec
from newton_calibration.isaaclab import tuning


def test_timing_contract_changes_resume_fingerprint(monkeypatch):
    plan = {"optimizer": {"generations": 4}, "environment": {"dt": 0.01}}
    options = {
        "optimizer_name": "fixture",
        "optimizer_version": "1",
        "optimizer_provider": "test",
        "initialization": SimpleNamespace(population=4, seed=1, options={}),
    }
    current = tuning._fit_execution_fingerprint(plan, **options)
    monkeypatch.setattr(tuning, "SAMPLE_TIMING_CONTRACT", "old-after-step-mislabeled-at-t0")
    assert tuning._fit_execution_fingerprint(plan, **options) != current


def episode(samples=4):
    return SimpleNamespace(
        name="timing-fixture",
        time_s=np.arange(samples) * 0.01,
        command_q=np.ones((samples, 1)),
        actual_q=np.zeros((samples, 1)),
        actual_dq=np.zeros((samples, 1)),
    )


def test_analytic_records_initial_then_correct_number_of_transitions():
    env = EnvironmentSpec(
        adapter="analytic",
        asset_path="fixture.usd",
        dt=0.01,
        profile_schema="articulation-profile/v1",
        joint_map={"a": "a"},
        joint_groups={"arm": ("a",)},
        base_stiffness=1.0,
        base_damping=0.0,
        base_effort_limit=10.0,
        analytic_inertia_by_joint={"a": 1.0},
    )
    runtime = AnalyticPDReplayRuntime(env)
    q, dq = runtime._rollout({}, episode())
    assert q[0, 0] == 0 and dq[0, 0] == 0
    assert q[1, 0] == pytest.approx(0.0001)
    assert dq[1, 0] == pytest.approx(0.01)
    one_q, one_dq = runtime._rollout({}, episode(1))
    assert one_q[0, 0] == 0 and one_dq[0, 0] == 0


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


@pytest.mark.parametrize("samples,delay", [(1, 0.0), (4, 0.0), (4, 0.01)])
def test_actual_newton_loop_with_fake_step_obeys_sample_clock(samples, delay):
    # Exercise the actual adapter loop with a deterministic in-memory fake;
    # deliberately no Isaac Lab import/GPU or physical response claim.
    runtime = IsaacLabNewtonRuntime.__new__(IsaacLabNewtonRuntime)
    runtime.environment = SimpleNamespace(device="cpu", dt=0.01)
    runtime.joint_layout = SimpleNamespace(logical_names=["a"])
    runtime.torch = SimpleNamespace(
        float32=None, as_tensor=lambda data, **kw: Tensor(data), zeros_like=lambda t: Tensor(np.zeros_like(t.data))
    )
    runtime._canonical_root_pose = runtime._canonical_root_velocity = None
    runtime._canonical_joint_position = runtime._canonical_joint_velocity = None
    runtime.env_ids_tensor = None
    runtime.joint_ids_tensor = [0]
    runtime._apply_candidate = runtime._verify_actuator_readback = lambda candidate: None
    runtime.residual = None
    position, velocity = Tensor([[0]]), Tensor([[0]])
    sent, steps = [], []

    def step():
        steps.append(1)
        position.data[:] += 1  # deterministic sentinel: state[k] must equal k

    runtime.robot = SimpleNamespace(
        reset=lambda: None,
        write_root_pose_to_sim_index=lambda **kw: None,
        write_root_velocity_to_sim_index=lambda **kw: None,
        write_joint_state_to_sim_index=lambda **kw: None,
        set_joint_effort_target_index=lambda **kw: None,
        set_joint_position_target_index=lambda **kw: sent.append(kw["target"].data.copy()),
        write_data_to_sim=lambda: None,
        update=lambda dt: None,
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=position), joint_vel=SimpleNamespace(torch=velocity)),
    )
    runtime.sim = SimpleNamespace(step=step)
    source = episode(samples)
    source.command_q = np.arange(samples, dtype=float)[:, None]
    q, dq, stable = runtime._rollout({"command_delay_s": delay}, source)
    np.testing.assert_array_equal(q[:, 0], np.arange(samples))
    np.testing.assert_array_equal(dq, np.zeros((samples, 1)))
    assert len(steps) == max(0, samples - 1) and stable
    for index, command in enumerate(sent):
        assert command[0, 0] == max(0, index - round(delay / 0.01))
