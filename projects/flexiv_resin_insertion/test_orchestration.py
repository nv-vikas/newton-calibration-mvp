"""CPU tests for the detached workflow's non-physics contracts."""

import json
from pathlib import Path

import pytest
from supervise import command, qualification_is_current, read_status


def test_command_is_argument_list_and_has_no_hardware_target():
    result = command(Path("/task/run.py"), Path("/work/assets with spaces"), Path("/work/run"), "train", 32, 12)
    assert result[result.index("--prepared") + 1] == "/work/assets with spaces"
    assert result[result.index("--num_envs") + 1] == "32"
    assert "--resume" in result
    assert "--headless" in result
    assert "flexivrdk" not in " ".join(result)


@pytest.mark.parametrize("content", [None, "{unfinished"])
def test_missing_or_incomplete_status_is_not_success(tmp_path, content):
    path = tmp_path / "status.json"
    if content is not None:
        path.write_text(content)
    assert not read_status(path).get("passed")


def test_failed_qualification_cannot_be_confused_with_success(tmp_path):
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps({"passed": False, "successful_envs": []}))
    assert read_status(path)["passed"] is False


def test_recipe_declares_material_assumptions_and_bounded_training():
    config = json.loads(Path(__file__).with_name("experiment.json").read_text())
    assert "unmeasured" in config["resin"]["basis"]
    assert config["training"]["max_wall_hours"] <= 48
    assert config["training"]["max_restarts"] <= 2
    assert config["geometry"]["peg_diameter_m"] < config["geometry"]["bore_diameter_m"]


@pytest.mark.parametrize(
    "state,updated,fingerprint,expected",
    [
        ("qualified", 101, "current", True),
        ("failed", 101, "current", False),
        ("qualified", 99, "current", False),
        ("qualified", 101, "different", False),
    ],
)
def test_stale_success_cannot_start_training(tmp_path, state, updated, fingerprint, expected):
    (tmp_path / "qualification.json").write_text(json.dumps({"passed": True, "experiment_fingerprint": "current"}))
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "state": state,
                "updated_unix": updated,
                "experiment_fingerprint": fingerprint,
            }
        )
    )
    assert qualification_is_current(tmp_path, 100) is expected
