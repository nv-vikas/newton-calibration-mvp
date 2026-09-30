"""Product conformance tests. The test double is never labeled as actual Newton evidence."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from newton_calibration.core.io import sha256_file
from newton_calibration.qualification import Gates, QualificationJob, Recipe, Settings
from newton_calibration.qualification.contracts import digest
from newton_calibration.qualification.gates import geometry_gate
from newton_calibration.qualification.runner import unseal


class Worker:
    def __init__(self, paths):
        self.paths = paths
        self.calls = []
        self.fail_once = None
        self.geometry_ok = True
        self.probes_ok = True
        self.controller_ok = True
        self.policy_ok = True
        self.runtime = "pinned-test-runtime"
        self.no_video = False
        self.changed_settings = False
        self.refinement_drift = 0
        self.require_small_margin = False
        self.tolerance_floor = 1e-12

    def describe(self):
        return {
            "runtime": "isaaclab_newton",
            "source": "test_double",
            "identity": self.runtime,
            "capabilities": ["geometry", "probes", "controlled_insertion", "frozen_policy"],
            "input_files": list(self.paths.values()),
            "parameter_bounds": {
                "dt_s": [1 / 120000, 1 / 120],
                "margin_m": [0, 0.001],
                "gap_m": [0, 0.001],
                "iterations": [1, 1000],
                "tolerance": [self.tolerance_floor, 0.001],
            },
            "trial_fingerprints": {name: digest(name) for name in ["dev-a", "dev-b", "held-a", "held-b"]},
        }

    def execute(self, request, output_dir):
        op = request["operation"]
        self.calls.append(copy.deepcopy(request))
        if op == self.fail_once:
            self.fail_once = None
            raise RuntimeError("interrupted fake worker")
        result = {
            "source": "test_double",
            "input_hashes": request["input_hashes"],
            "applied_settings": request["settings"],
            "policy_hz": request["policy_hz"],
            "artifacts": [],
        }
        if self.changed_settings:
            result["applied_settings"] = {"dt_s": 99}
        if op == "geometry":
            return {
                **result,
                "minimum_radial_clearance_m": 25e-6,
                "geometry_error_bound_m": 1e-6,
                "coverage": "full_insertion_depth",
                "collision_enabled": True,
                "bore_preserved": self.geometry_ok,
            }
        depth = 0.039
        if op == "probes":
            if not self.probes_ok or (self.require_small_margin and request["settings"]["margin_m"] > 5e-6):
                depth = 0.0065
            if request["settings"]["dt_s"] < 1 / 960:
                depth += self.refinement_drift
        if op == "controlled_insertion" and not self.controller_ok:
            depth = 0.002
        if op == "frozen_policy" and not self.policy_ok:
            depth = 0.002
        result["trials"] = [
            {
                "id": name,
                "valid": True,
                "depth_m": depth,
                "hold_s": 0.3,
                "max_wall_penetration_m": 1e-6,
                "max_grasp_displacement_m": 0.001,
                "rim_depth_m": 1e-5,
                "policy_success": self.policy_ok,
            }
            for name in request["trials"]
        ]
        if op == "controlled_insertion":
            result["diagnostic_phases"] = {
                phase: {
                    "completed_trial_ids": request["trials"],
                    "max_wall_penetration_m": 0.0,
                    "max_grasp_displacement_m": 0.001,
                }
                for phase in ("hold", "free_motion")
            }
        path = Path(output_dir) / "trace.json"
        path.write_text(json.dumps(result))
        result["artifacts"].append(path.name)
        if request["video"] and not self.no_video:
            # Bytes are a contract artifact for this explicitly labeled test double, not a fabricated demo.
            (Path(output_dir) / "test-only.mp4").write_bytes(b"test-double-not-a-video")
            result["artifacts"].append("test-only.mp4")
        return result


@pytest.fixture
def setup(tmp_path):
    inputs = {}
    for name in ["asset", "policy", "controller", "scene", "trial_bank"]:
        path = tmp_path / f"{name}.txt"
        path.write_text(name)
        inputs[name] = str(path)
    recipe = Recipe("tight fit", inputs, ("dev-a", "dev-b"), ("held-a", "held-b"))
    worker = Worker(inputs)
    return recipe, worker, tmp_path / "job"


def test_refinement_respects_declared_runtime_tolerance_floor(setup):
    recipe, worker, root = setup
    worker.tolerance_floor = 1e-6
    result = QualificationJob.create(recipe, worker, root).run()
    assert result["state"] == "complete"
    refined = [c for c in worker.calls if c["operation"] == "probes" and c["settings"]["dt_s"] < 1 / 960]
    assert len(refined) == 1
    assert refined[0]["settings"]["tolerance"] == 1e-6
    assert refined[0]["settings"]["iterations"] == 200


def test_explicit_recipe_below_runtime_floor_rejected_before_execution(setup):
    recipe, worker, root = setup
    worker.tolerance_floor = 1e-6
    recipe = replace(recipe, baseline=Settings(tolerance=1e-7))
    with pytest.raises(ValueError, match="Unsupported setting tolerance"):
        QualificationJob.create(recipe, worker, root)
    assert not worker.calls
    assert not root.exists()


def test_four_stages_and_frozen_evaluation(setup):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    assert not worker.calls
    assert job.status()["next_stage"] == "geometry"
    with pytest.raises(ValueError, match="skip/reorder"):
        job.advance(expected_stage="frozen_policy")
    result = job.run()
    assert result["state"] == "complete"
    assert result["source"] == "test_double"
    assert result["physics_qualified"] and result["frozen_policy_passed"]
    assert not result["real_world_calibrated"] and not result["retrained"]
    assert all(c["trials"] == list(recipe.validation_trials) for c in worker.calls if c["operation"] == "frozen_policy")
    assert all(not set(c["trials"]) & set(recipe.validation_trials) for c in worker.calls if c["operation"] == "probes")
    assert (root / "qualified_simulation.json").exists()
    count = len(worker.calls)
    assert QualificationJob(root, worker).run() == result
    assert len(worker.calls) == count


@pytest.mark.parametrize("which", ["geometry", "probes", "controller", "policy"])
def test_failure_gates_stop_later_stages(setup, which):
    recipe, worker, root = setup
    setattr(worker, f"{which}_ok", False)
    result = QualificationJob.create(recipe, worker, root).run()
    assert result["state"] == "blocked"
    if which != "policy":
        assert result["frozen_policy_passed"] is None
        assert not any(c["operation"] == "frozen_policy" for c in worker.calls)
    else:
        assert result["physics_qualified"]
        assert result["frozen_policy_passed"] is False


def test_resume_after_worker_failure_does_not_repeat_completed_experiments(setup):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    worker.fail_once = "controlled_insertion"
    with pytest.raises(RuntimeError, match="interrupted"):
        job.run()
    count = sum(c["operation"] == "probes" for c in worker.calls)
    assert list(root.glob("experiments/controlled/attempt-*/failure.json"))
    result = QualificationJob(root, worker).run()
    assert result["state"] == "complete"
    assert sum(c["operation"] == "probes" for c in worker.calls) == count
    assert len(list(root.glob("experiments/controlled/attempt-*"))) == 2


@pytest.mark.parametrize("field", ["asset", "policy", "controller", "scene", "trial_bank"])
def test_input_drift_blocks_resume(setup, field):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    job.advance()
    Path(recipe.inputs[field]).write_text("changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        job.run()


def test_runtime_drift_blocks_resume(setup):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    worker.runtime = "changed-runtime"
    with pytest.raises(ValueError, match="identity changed"):
        job.run()


def test_artifact_tampering_is_detected(setup):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    job.run()
    next(root.glob("experiments/solver-000/attempt-*/trace.json")).write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        job.status()


def test_plan_and_stage_hashes(setup):
    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    job.advance()
    path = root / "stages/01-geometry.json"
    data = json.loads(path.read_text())
    data["payload"]["passed"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Modified qualification record"):
        job.status()


def test_partial_settings_or_renamed_holdouts_rejected(setup):
    recipe, worker, root = setup
    with pytest.raises(ValueError, match="disjoint"):
        replace(recipe, validation_trials=("dev-a",))
    original = worker.describe
    worker.describe = lambda: {
        **original(),
        "trial_fingerprints": {i: "same-content" for i in original()["trial_fingerprints"]},
    }
    with pytest.raises(ValueError, match="Trial content"):
        QualificationJob.create(recipe, worker, root)


def test_sweep_selects_first_physics_qualified_candidate(setup):
    recipe, worker, root = setup
    worker.require_small_margin = True
    recipe = replace(recipe, candidates=(Settings(margin_m=10e-6), Settings(margin_m=5e-6)))
    job = QualificationJob.create(recipe, worker, root)
    result = job.run()
    assert result["state"] == "complete"
    selection = unseal(root / "stages/02-solver.json")
    assert len(selection["candidates"]) == 3
    assert selection["selected_settings"]["margin_m"] == 5e-6


def test_refinement_must_converge(setup):
    recipe, worker, root = setup
    worker.refinement_drift = 0.001
    result = QualificationJob.create(recipe, worker, root).run()
    assert result["state"] == "blocked"
    assert result["stages"][-1]["stage"] == "solver"


@pytest.mark.parametrize("option", ["no_video", "changed_settings"])
def test_missing_video_or_ignored_settings_fails_closed(setup, option):
    recipe, worker, root = setup
    setattr(worker, option, True)
    with pytest.raises(ValueError):
        QualificationJob.create(recipe, worker, root).run()


@pytest.mark.parametrize("key", ["repeatability_m", "convergence_m", "max_wall_penetration_m"])
def test_nonzero_tolerances(key):
    with pytest.raises(ValueError):
        Gates(**{key: 0})


def test_tight_fit_cannot_use_loose_wall_gate():
    data = {
        "minimum_radial_clearance_m": 25e-6,
        "geometry_error_bound_m": 1e-6,
        "coverage": "full_insertion_depth",
        "bore_preserved": True,
        "collision_enabled": True,
    }
    assert not geometry_gate(data, Gates(max_wall_penetration_m=200e-6))["passed"]


def test_policy_cadence_and_unknown_tunables(setup):
    recipe, worker, root = setup
    with pytest.raises(ValueError, match="integer decimation"):
        replace(recipe, baseline=Settings(dt_s=0.003))
    with pytest.raises(TypeError):
        Settings(friction=0)
    with pytest.raises(FileExistsError):
        QualificationJob.create(recipe, worker, root)
        QualificationJob.create(recipe, worker, root)


def test_missing_minjae_provider_is_not_substituted(setup):
    recipe, worker, root = setup
    recipe = replace(recipe, optimizer={"name": "minjae-not-installed.v1", "parameters": []})
    with pytest.raises(KeyError, match="Unknown optimizer"):
        QualificationJob.create(recipe, worker, root)


def test_optimizer_receives_only_development_physics_scores_and_resumes(setup):
    from newton_calibration.optimizers import register_optimizer

    received = []

    class Optimizer:
        generation = 0
        best = None

        def ask(self):
            return [{"margin_m": 5e-6}]

        def tell(self, candidates, scores):
            received.extend(scores)
            self.generation += 1

        def state_dict(self):
            return {"generation": self.generation}

        def load_state_dict(self, state):
            self.generation = state["generation"]

    register_optimizer("qualification-test-only", lambda _: Optimizer(), version="1", replace=True)
    recipe, worker, root = setup
    worker.require_small_margin = True
    recipe = replace(
        recipe,
        optimizer={
            "name": "qualification-test-only",
            "parameters": [
                {
                    "name": "margin_m",
                    "lower": 0,
                    "upper": 50e-6,
                    "initial": 25e-6,
                    "unit": "m",
                    "owner": "simulator",
                    "rationale": "test",
                },
            ],
        },
    )
    result = QualificationJob.create(recipe, worker, root).run()
    assert result["state"] == "complete"
    assert received == [0.0]
    assert sha256_file(recipe.inputs["policy"]) == unseal(root / "plan.json")["inputs"]["policy"]["sha256"]


def test_incomplete_free_motion_diagnostic_blocks_policy(setup):
    recipe, worker, root = setup
    original = worker.execute

    def execute(request, output_dir):
        result = original(request, output_dir)
        if request["operation"] == "controlled_insertion":
            result["diagnostic_phases"]["free_motion"]["completed_trial_ids"] = []
        return result

    worker.execute = execute
    result = QualificationJob.create(recipe, worker, root).run()
    assert result["state"] == "blocked"
    assert result["stages"][-1]["diagnostic_phases"]["free_motion"] is False
    assert not any(c["operation"] == "frozen_policy" for c in worker.calls)


def test_job_lock_prevents_two_orchestrators(setup):
    import fcntl

    recipe, worker, root = setup
    job = QualificationJob.create(recipe, worker, root)
    with (root / ".qualification.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            job.advance()


def test_agent_tools_are_guided_and_do_not_accept_executable_factories(setup):
    from dataclasses import asdict

    from newton_calibration.qualification import AgentTools

    recipe, worker, root = setup
    tools = AgentTools(worker, root)
    assert len(tools.describe()["tools"]) == 3
    with pytest.raises(ValueError, match="simple name"):
        tools.create(job_id="../escape", recipe=asdict(recipe))
    assert tools.create(job_id="test-job", recipe=asdict(recipe))["next_stage"] == "geometry"
    with pytest.raises(ValueError, match="skip/reorder"):
        tools.advance(job_id="test-job", expected_stage="frozen_policy")
    assert tools.advance(job_id="test-job", expected_stage="geometry")["next_stage"] == "solver"
