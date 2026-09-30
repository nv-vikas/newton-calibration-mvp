"""Isolated worker transport. Explicit trusted argv only; never shell/eval a recipe."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path

from newton_calibration.core.io import atomic_write_json


class CommandBackend:
    def __init__(self, argv: list[str], *, timeout_s: float = 900):
        if not argv or any(not isinstance(x, str) or not x for x in argv) or timeout_s <= 0:
            raise ValueError("Supply trusted worker argv and a positive timeout")
        self.argv = tuple(argv)
        self.timeout_s = timeout_s

    def _call(self, request, output_dir):
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(root / "request.json", request)
        with (root / "worker.log").open("xb") as log:
            process = subprocess.Popen(
                [*self.argv, "--request", str(root / "request.json"), "--output", str(root / "response.json")],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=min(self.timeout_s, request.get("timeout_s", self.timeout_s)))
            except BaseException:
                # Terminate only this invocation's process group, never training or unrelated workers.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if code != 0:
            raise RuntimeError(f"Qualification worker exited {code}; inspect {root / 'worker.log'}")
        if not (root / "response.json").is_file():
            raise RuntimeError(
                f"Worker exited without a completed result; inspect {root / 'worker.log'} and physics_failure.json"
            )
        return json.loads((root / "response.json").read_text())

    def describe(self):
        with tempfile.TemporaryDirectory(prefix="newton-qualification-describe-") as root:
            description = self._call({"operation": "describe"}, root)
        return {**description, "transport": {"argv": list(self.argv), "timeout_s": self.timeout_s}}

    def execute(self, request, output_dir):
        return self._call(request, output_dir)
