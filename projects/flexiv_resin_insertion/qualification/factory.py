"""Run inside the pinned Isaac Lab Docker image with an explicit trusted binding."""

import os
import sys
from pathlib import Path

from newton_calibration.qualification import CommandBackend


def create_backend():
    binding = Path(os.environ["NEWTON_QUALIFICATION_BINDING"]).resolve()
    return CommandBackend(
        [sys.executable, str(Path(__file__).with_name("worker.py")), "--binding", str(binding)], timeout_s=1800
    )
