import json
import subprocess
import sys

import pytest

from newton_calibration.qualification import CommandBackend


def test_shell_free_worker_transport(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        "import argparse,json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--output')\n"
        "a=p.parse_args(); r=json.loads(Path(a.request).read_text())\n"
        "Path(a.output).write_text(json.dumps({'echo': r}))\n"
    )
    backend = CommandBackend([sys.executable, str(script)])
    request = {"operation": "example", "data": "$(touch do-not-create); never executed"}
    result = backend.execute(request, tmp_path / "run with spaces")
    assert result == {"echo": request}
    assert json.loads((tmp_path / "run with spaces/request.json").read_text()) == request
    assert not (tmp_path / "do-not-create").exists()


def test_worker_timeout_retains_log(tmp_path):
    backend = CommandBackend(
        [sys.executable, "-c", "import time; print('starting', flush=True); time.sleep(20)"], timeout_s=0.2
    )
    with pytest.raises(subprocess.TimeoutExpired):
        backend.execute({"operation": "probes"}, tmp_path / "attempt")
    assert (tmp_path / "attempt/worker.log").exists()


def test_worker_error_never_becomes_success(tmp_path):
    backend = CommandBackend([sys.executable, "-c", "raise RuntimeError('worker failed')"])
    with pytest.raises(RuntimeError, match="exited"):
        backend.execute({"operation": "probes"}, tmp_path / "attempt")
    assert "worker failed" in (tmp_path / "attempt/worker.log").read_text()


def test_fast_exit_without_published_result_is_not_success(tmp_path):
    backend = CommandBackend([sys.executable, "-c", "pass"])
    with pytest.raises(RuntimeError, match="without a completed result"):
        backend.execute({"operation": "probes"}, tmp_path / "attempt")
