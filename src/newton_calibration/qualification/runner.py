"""Bounded four-stage qualification with immutable plans and resumable experiment receipts."""

from __future__ import annotations

import copy
import fcntl
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from newton_calibration.core.io import atomic_write_json, atomic_write_text, sha256_file, utc_now
from newton_calibration.core.models import ParameterSpec
from newton_calibration.optimizers import OptimizerInit, get_optimizer_registration, validate_candidates

from .contracts import Backend, Recipe, Settings, digest
from .gates import compare_probes, controlled_gate, geometry_gate, insertion_gate, probes_gate

STAGES = ("geometry", "solver", "controlled_insertion", "frozen_policy")


def read(path):
    return json.loads(Path(path).read_text())


def seal(path, payload):
    atomic_write_json(path, {"payload": payload, "sha256": digest(payload)})


def unseal(path):
    value = read(path)
    if value["sha256"] != digest(value["payload"]):
        raise ValueError(f"Modified qualification record: {path}")
    return value["payload"]


class QualificationJob:
    def __init__(self, directory, backend: Backend):
        self.root = Path(directory).resolve()
        self.backend = backend
        self.plan = unseal(self.root / "plan.json")
        self.recipe = Recipe.from_dict(self.plan["recipe"])

    @classmethod
    def create(cls, recipe: Recipe, backend: Backend, directory):
        # Snapshot nested mutable values; callers cannot mutate a locked plan in place.
        recipe = Recipe.from_dict(copy.deepcopy(asdict(recipe)))
        description = backend.describe()
        required = {"geometry", "probes", "controlled_insertion", "frozen_policy"}
        if not required <= set(description["capabilities"]):
            raise ValueError("Worker lacks one or more qualification operations")
        if description["source"] not in {"simulation", "test_double"} or description["runtime"] != "isaaclab_newton":
            raise ValueError("Qualification requires an Isaac Lab/Newton simulation worker")
        if not description.get("identity"):
            raise ValueError("Worker must identify its runtime, code and controller implementation")
        inputs = {key: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for key, p in recipe.inputs.items()}
        # The adapter must enumerate its complete dependency closure, not only the root USD.
        for path in description["input_files"]:
            if str(Path(path).resolve()) not in {entry["path"] for entry in inputs.values()}:
                raise ValueError(f"Worker input not pinned in recipe: {path}")
        fingerprints = description["trial_fingerprints"]
        dev = {fingerprints[i] for i in recipe.development_trials}
        held = {fingerprints[i] for i in recipe.validation_trials}
        if len(dev) != len(recipe.development_trials) or len(held) != len(recipe.validation_trials) or dev & held:
            raise ValueError("Trial content is duplicated, renamed or leaks between development and validation")
        root = Path(directory).resolve()
        if root.exists():
            raise FileExistsError("Create a new revision directory or resume the existing job")
        optimizer_identity = None
        if recipe.optimizer:
            config = recipe.optimizer
            if set(config) - {"name", "parameters", "options", "seed"}:
                raise ValueError("Unknown optimizer configuration keys")
            registration = get_optimizer_registration(config["name"])
            parameters = tuple(ParameterSpec(**p) for p in config["parameters"])
            # Discrete choices are explicit sweep entries, not silently rounded continuous proposals.
            if not parameters or len({p.name for p in parameters}) != len(parameters):
                raise ValueError("Require unique optimizer parameters")
            for p in parameters:
                if p.name not in {"margin_m", "gap_m", "tolerance"}:
                    raise ValueError(
                        "Optimizer supports continuous margin_m, gap_m, tolerance; use sweeps for dt/iterations"
                    )
                lower, upper = description["parameter_bounds"][p.name]
                if p.lower < lower or p.upper > upper:
                    raise ValueError("Optimizer bounds exceed the worker's supported bounds")
            optimizer_identity = {
                "name": registration.name,
                "version": registration.version,
                "provider": registration.provider,
            }
        plan = {
            "schema": "newton.qualification/locked-plan-v1",
            "created_at": utc_now(),
            "recipe": asdict(recipe),
            "inputs": inputs,
            "backend": description,
            "optimizer": optimizer_identity,
            "scope": "simulation qualification only; no hardware, retraining or real-world calibration",
        }
        # Validate even refinement settings before any physics run.
        bounds = description["parameter_bounds"]
        for setting in (recipe.baseline, *recipe.candidates):
            for value in (setting, setting.refined(tolerance_floor=bounds["tolerance"][0])):
                for key, number in asdict(value).items():
                    if key not in bounds or not bounds[key][0] <= number <= bounds[key][1]:
                        raise ValueError(f"Unsupported setting {key}={number}")
        root.mkdir(parents=True)
        seal(root / "plan.json", plan)
        job = cls(root, backend)
        job._publish([])
        return job

    @contextmanager
    def _locked(self):
        with (self.root / ".qualification.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another worker already owns this qualification job") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _verify(self):
        if unseal(self.root / "plan.json") != self.plan:
            raise ValueError("Plan changed; create a new revision")
        for entry in self.plan["inputs"].values():
            if sha256_file(entry["path"]) != entry["sha256"]:
                raise ValueError(f"Pinned input changed: {entry['path']}")
        if self.backend.describe() != self.plan["backend"]:
            raise ValueError("Runtime/provider/controller/trial identity changed; create a new baseline revision")
        if self.recipe.optimizer:
            registration = get_optimizer_registration(self.recipe.optimizer["name"])
            if self.plan["optimizer"] != {
                "name": registration.name,
                "version": registration.version,
                "provider": registration.provider,
            }:
                raise ValueError("Optimizer provider changed; create a new revision")

    def _operation(self, key, operation, settings=None, trials=(), *, video=False):
        self._verify()
        request = {
            "schema": "newton.qualification/experiment-v1",
            "operation": operation,
            "plan_sha256": digest(self.plan),
            "settings": asdict(settings) if settings else None,
            "trials": list(trials),
            "policy_hz": self.recipe.policy_hz,
            "gates": asdict(self.recipe.gates),
            "video": video,
            "timeout_s": self.recipe.timeout_s,
            "input_hashes": {k: e["sha256"] for k, e in self.plan["inputs"].items()},
        }
        folder = self.root / "experiments" / key
        receipt = folder / "receipt.json"
        if receipt.exists():
            record = unseal(receipt)
            if record["request"] != request:
                raise ValueError("Recorded experiment does not match the resumed request")
            self._verify_artifacts(record)
            return record["result"]
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))):04d}"
        attempt.mkdir()
        atomic_write_json(attempt / "invocation.json", request)
        seal(
            self.root / "progress.json",
            {
                "state": "running",
                "experiment": key,
                "operation": operation,
                "attempt": str(attempt.relative_to(self.root)),
                "started_at": utc_now(),
                "plan_sha256": digest(self.plan),
            },
        )
        try:
            result = self.backend.execute(copy.deepcopy(request), str(attempt))
            self._verify()
            # Echo/readback is mandatory. A backend must fail on unsupported settings, never silently ignore them.
            if result["applied_settings"] != request["settings"] or result["policy_hz"] != self.recipe.policy_hz:
                raise ValueError("Worker settings/readback or policy cadence differs from locked request")
            if result["source"] != self.plan["backend"]["source"] or result["input_hashes"] != request["input_hashes"]:
                raise ValueError("Worker provenance mismatch")
            artifacts = {}
            for relative in result["artifacts"]:
                path = (attempt / relative).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or path.is_symlink():
                    raise ValueError("Worker artifact must be a regular file inside its attempt directory")
                artifacts[str(path.relative_to(self.root))] = sha256_file(path)
            if operation != "geometry" and not artifacts:
                raise ValueError("Physics experiments require trace artifacts")
            if video and not any(p.endswith(".mp4") for p in artifacts):
                raise ValueError("Requested video was not produced; do not claim a complete recorded evaluation")
            record = {"request": request, "result": result, "artifacts": artifacts, "completed_at": utc_now()}
            seal(receipt, record)
            seal(
                self.root / "progress.json",
                {
                    "state": "completed",
                    "experiment": key,
                    "operation": operation,
                    "attempt": str(attempt.relative_to(self.root)),
                    "completed_at": utc_now(),
                    "plan_sha256": digest(self.plan),
                },
            )
            return result
        except BaseException as exc:
            atomic_write_json(
                attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "at": utc_now()}
            )
            raise

    def _verify_artifacts(self, record):
        for relative, expected in record["artifacts"].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or sha256_file(path) != expected:
                raise ValueError(f"Qualification artifact changed: {relative}")

    def _stages(self):
        results = []
        for index, stage in enumerate(STAGES):
            path = self.root / "stages" / f"{index + 1:02d}-{stage}.json"
            if path.exists():
                if len(results) != index:
                    raise ValueError("Non-contiguous qualification stages")
                result = unseal(path)
                if result["stage"] != stage or result["plan_sha256"] != digest(self.plan):
                    raise ValueError("Mixed qualification stage record")
                results.append(result)
        # Verify receipts even when a stage has already finished.
        for path in sorted((self.root / "experiments").glob("*/receipt.json")):
            record = unseal(path)
            if record["request"]["plan_sha256"] != digest(self.plan):
                raise ValueError("Mixed experiment plan")
            self._verify_artifacts(record)
        return results

    def status(self):
        self._verify()
        result = self._summary(self._stages())
        if result["state"] == "ready":
            with (self.root / ".qualification.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    result.update(state="running", next_action="Wait for the active worker; inspect progress/logs")
                else:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            progress = self.root / "progress.json"
            if progress.exists():
                current = unseal(progress)
                if current["plan_sha256"] != digest(self.plan):
                    raise ValueError("Progress belongs to another plan")
                result["last_experiment"] = current
        return result

    def _summary(self, stages):
        failed = next((s for s in stages if not s["passed"]), None)
        state = "blocked" if failed else "complete" if len(stages) == 4 else "ready"
        return {
            "state": state,
            "source": self.plan["backend"]["source"],
            "next_stage": STAGES[len(stages)] if state == "ready" else None,
            "next_action": f"Inspect {failed['stage']} results; changes require a new recipe revision"
            if failed
            else "Qualification finished; read physics and policy outcomes separately"
            if state == "complete"
            else f"Run {STAGES[len(stages)]} through the same toolkit API",
            "physics_qualified": len(stages) >= 3 and all(s["passed"] for s in stages[:3]),
            "frozen_policy_passed": stages[3]["passed"] if len(stages) == 4 else None,
            "real_world_calibrated": False,
            "retrained": False,
            "stages": stages,
            "scope": self.plan["scope"],
        }

    def _publish(self, stages):
        summary = self._summary(stages)
        atomic_write_json(self.root / "status.json", summary)
        lines = [f"# {self.recipe.name}", "", self.plan["scope"], "", f"State: **{summary['state']}**", ""]
        for stage in stages:
            lines.append(f"- {stage['stage']}: {'passed' if stage['passed'] else 'blocked/failed'}")
        lines += ["", f"Next: {summary['next_action']}", "", "Simulation assumptions are not real measurements."]
        atomic_write_text(self.root / "REPORT.md", "\n".join(lines) + "\n")
        if len(stages) >= 3 and all(s["passed"] for s in stages[:3]):
            seal(
                self.root / "qualified_simulation.json",
                {
                    "schema": "newton.qualification/scoped-numerical-configuration-v1",
                    "plan_sha256": digest(self.plan),
                    "settings": stages[1]["selected_settings"],
                    "policy_result": summary["frozen_policy_passed"],
                    "source": summary["source"],
                    "activation": "simulation-only; not a real-data calibration package",
                    "inputs": self.plan["inputs"],
                },
            )
        return summary

    def advance(self, *, expected_stage=None):
        """Execute exactly one guarded stage. Re-enter after interruptions to reuse committed experiments."""
        with self._locked():
            self._verify()
            stages = self._stages()
            summary = self._summary(stages)
            if summary["state"] != "ready":
                return self._publish(stages)
            stage = summary["next_stage"]
            if expected_stage is not None and stage != expected_stage:
                raise ValueError(f"Cannot skip/reorder stages: next is {stage}")
            try:
                result = getattr(self, f"_{stage}")(stages)
                result.update(stage=stage, plan_sha256=digest(self.plan))
                seal(self.root / "stages" / f"{len(stages) + 1:02d}-{stage}.json", result)
                return self._publish([*stages, result])
            except BaseException as exc:
                atomic_write_json(
                    self.root / "status.json",
                    {
                        **summary,
                        "state": "execution_failed",
                        "error": str(exc),
                        "next_action": "Correct worker availability and resume; input changes require a new revision",
                    },
                )
                raise

    def run(self):
        while True:
            result = self.advance()
            if result["state"] != "ready":
                return result

    def _geometry(self, stages):
        result = self._operation("geometry", "geometry")
        return {**geometry_gate(result, self.recipe.gates), "measurement": result}

    def _qualify_candidate(self, index, settings):
        settings.validate_rate(self.recipe.policy_hz)
        floor = self.plan["backend"]["parameter_bounds"]["tolerance"][0]
        refined = settings.refined(tolerance_floor=floor)
        for config in (settings, refined):
            for key, number in asdict(config).items():
                lo, hi = self.plan["backend"]["parameter_bounds"][key]
                if not lo <= number <= hi:
                    raise ValueError(f"Candidate/refinement exceeds supported {key} bounds")
        ids, gates = self.recipe.development_trials, self.recipe.gates
        prefix = f"solver-{index:03d}"
        first = self._operation(prefix, "probes", settings, ids)
        first_gate = probes_gate(first, ids, gates)
        result = {
            "settings": asdict(settings),
            "refinement_settings": asdict(refined),
            "refinement_tolerance_at_runtime_floor": refined.tolerance == floor,
            "probe_gate": first_gate,
            "passed": False,
        }
        if not first_gate["passed"]:
            return result
        repeat = self._operation(prefix + "-repeat", "probes", settings, ids)
        fine = self._operation(prefix + "-refined", "probes", refined, ids)
        result.update(
            repeat_gate=probes_gate(repeat, ids, gates),
            refined_gate=probes_gate(fine, ids, gates),
            repeatability=compare_probes(first, repeat, ids, gates.repeatability_m),
            convergence=compare_probes(first, fine, ids, gates.convergence_m),
        )
        result["passed"] = all(
            result[k]["passed"]
            for k in (
                "probe_gate",
                "repeat_gate",
                "refined_gate",
                "repeatability",
                "convergence",
            )
        )
        return result

    def _solver(self, stages):
        results = [self._qualify_candidate(0, self.recipe.baseline)]
        optimizer = None
        if self.recipe.optimizer:
            config = self.recipe.optimizer
            parameters = tuple(ParameterSpec(**p) for p in config["parameters"])
            optimizer = get_optimizer_registration(config["name"]).create(
                OptimizerInit(
                    parameters,
                    population=1,
                    seed=config.get("seed", 0),
                    options=config.get("options", {}),
                )
            )
        for index in range(self.recipe.max_candidates):
            if results[-1]["passed"]:
                break
            if optimizer:
                # Each proposal is journaled before physics; optimizer post-tell state is committed separately.
                path = self.root / "search" / f"{index:03d}.json"
                proposal_path = self.root / "search" / f"{index:03d}-proposal.json"
                if proposal_path.exists():
                    proposal = unseal(proposal_path)
                    optimizer.load_state_dict(proposal["after_ask_state"])
                    candidates = validate_candidates(proposal["candidates"], parameters, expected_count=1)
                else:
                    candidates = validate_candidates(optimizer.ask(), parameters, expected_count=1)
                    seal(proposal_path, {"candidates": candidates, "after_ask_state": optimizer.state_dict()})
                settings = Settings(**{**asdict(self.recipe.baseline), **candidates[0]})
                result = self._qualify_candidate(index + 1, settings)
                # Physics validity/convergence only. No policy rewards or validation trials reach the optimizer.
                score = 0.0 if result["passed"] else 1.0 + len(result["probe_gate"]["reasons"])
                if path.exists():
                    committed = unseal(path)
                    if committed["candidates"] != candidates or committed["score"] != score:
                        raise ValueError("Optimizer journal no longer matches physics receipts")
                    optimizer.load_state_dict(committed["state"])
                else:
                    optimizer.tell(candidates, [score])
                    seal(path, {"candidates": candidates, "score": score, "state": optimizer.state_dict()})
                results.append(result)
            else:
                if index >= len(self.recipe.candidates):
                    break
                results.append(self._qualify_candidate(index + 1, self.recipe.candidates[index]))
        winner = next((r for r in results if r["passed"]), None)
        return {
            "passed": winner is not None,
            "candidates": results,
            "selected_settings": winner["settings"] if winner else None,
            "stop_reason": "first_qualified_configuration" if winner else "declared_search_exhausted",
            "not_claimed": "global optimum or every possible solver setting explored",
        }

    def _controlled_insertion(self, stages):
        settings = Settings(**stages[1]["selected_settings"])
        result = self._operation(
            "controlled",
            "controlled_insertion",
            settings,
            self.recipe.development_trials,
            video=self.recipe.record_video,
        )
        return {
            **controlled_gate(result, self.recipe.development_trials, self.recipe.gates),
            "measurements": result,
            "controller": "scripted environment diagnostic; not a learned policy",
        }

    def _frozen_policy(self, stages):
        settings = Settings(**stages[1]["selected_settings"])
        ids = self.recipe.validation_trials
        # Baseline can be invalid: still record it, but never promote its apparent success over precision gates.
        before = self._operation(
            "policy-before", "frozen_policy", self.recipe.baseline, ids, video=self.recipe.record_video
        )
        after = self._operation("policy-after", "frozen_policy", settings, ids, video=self.recipe.record_video)
        a = insertion_gate(before, ids, self.recipe.gates, policy=True)
        b = insertion_gate(after, ids, self.recipe.gates, policy=True)
        return {
            "passed": b["passed"],
            "before": a,
            "after": b,
            "policy_sha256": self.plan["inputs"]["policy"]["sha256"],
            "retrained": False,
            "success_difference": b["successes"] - a["successes"],
            "interpretation": "matched simulated trials; no guarantee of real-world transfer",
        }
