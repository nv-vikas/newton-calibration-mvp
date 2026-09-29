"""Single-writer, revisioned sessions. Fits resume using the existing fit journal."""

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from newton_calibration.core.io import utc_now, write_json


def read(path):
    def constant(value):
        raise ValueError(f"Nonfinite JSON is not supported: {value}")

    value = json.loads(Path(path).read_text(), parse_constant=constant)
    json.dumps(value, allow_nan=False)
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def locked(directory):
    directory = Path(directory).resolve()
    if not (directory / "session.json").is_file():
        raise FileNotFoundError(f"No guided session at {directory}")
    with (directory / ".session.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This session already has an active writer; inspect status or wait") from exc
        try:
            session = read(directory / "session.json")
            if session.get("schema") != "newton.guided-session/v1":
                raise ValueError("Unsupported session schema")
            yield directory, session
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save(directory, session, event):
    session["updated_at"] = utc_now()
    session.setdefault("events", []).append(
        {"at": session["updated_at"], "event": event, "revision": session["revision"], "state": session["state"]}
    )
    write_json(directory / "session.json", session)


def remember(directory, session, stage, path):
    path = Path(path).resolve()
    relative = path.relative_to(directory).as_posix()
    session["artifacts"][stage] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save(directory, session, f"{stage}:recorded")


def artifact(directory, session, stage):
    ref = session["artifacts"][stage]
    path = (directory / ref["path"]).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"Invalid artifact path for {stage}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
        raise ValueError(f"Saved {stage} record changed; create a new revision")
    return read(path)
