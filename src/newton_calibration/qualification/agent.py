"""Small framework-neutral tool surface for Minjae or another orchestrating agent."""

import re
from pathlib import Path

from .contracts import Recipe
from .runner import STAGES, QualificationJob


class AgentTools:
    """The operator binds a trusted backend; model-provided arguments cannot select code.

    This is an integration surface, not an installed LLM, scheduling service or
    hardware agent. The host agent owns authentication, approvals and scheduling.
    """

    def __init__(self, backend, jobs_root):
        self.backend = backend
        self.root = Path(jobs_root).resolve()

    def _path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", job_id):
            raise ValueError("Job ID must be a simple name, not a file path")
        path = self.root / job_id
        if path.resolve().parent != self.root:
            raise ValueError("Job path escapes the configured job root")
        return path

    def describe(self):
        return {
            "capabilities": self.backend.describe(),
            "stages": list(STAGES),
            "tools": [
                {
                    "name": "qualification_create",
                    "arguments": {"job_id": "string", "recipe": "Recipe JSON"},
                    "purpose": "Lock a new recipe and input revision; do not start physics",
                },
                {
                    "name": "qualification_status",
                    "arguments": {"job_id": "string"},
                    "purpose": "Read verified state, next action, separate physics and frozen-policy outcomes",
                },
                {
                    "name": "qualification_advance",
                    "arguments": {"job_id": "string", "expected_stage": list(STAGES)},
                    "purpose": "Run only the next guarded stage; reuse committed experiments when resuming",
                },
            ],
            "rules": [
                "No training, hardware operation, arbitrary shell, geometry/material edits or controller retuning",
                "Do not change gates or use reserved policy trials to select solver settings",
                "Blocked physics gates stop progression; report the reason rather than bypassing the gate",
                "Changes to inputs/settings/recipe require a new revision and baseline",
                "Simulation-only qualification is not evidence of real-world calibration or transfer",
            ],
        }

    def create(self, *, job_id, recipe):
        return QualificationJob.create(Recipe.from_dict(recipe), self.backend, self._path(job_id)).status()

    def status(self, *, job_id):
        return QualificationJob(self._path(job_id), self.backend).status()

    def advance(self, *, job_id, expected_stage):
        if expected_stage not in STAGES:
            raise ValueError("Select a declared stage")
        return QualificationJob(self._path(job_id), self.backend).advance(expected_stage=expected_stage)
