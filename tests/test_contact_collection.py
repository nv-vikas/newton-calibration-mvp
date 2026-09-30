import json
from copy import deepcopy
from pathlib import Path

import pytest

from newton_calibration.collection.contact import SIGNALS, load_collection_recipe, verify_collection_bundle
from newton_calibration.guided import advance, provide, run, start
from newton_calibration.guided.catalog import CONTACT_RECIPES, get_recipe
from newton_calibration.guided.cli import describe
from newton_calibration.isaaclab import tuning


@pytest.fixture
def asset(tmp_path):
    path = tmp_path / "robot.usda"
    path.write_text('#usda 1.0\ndef Xform "Robot" {}\n')
    return path


@pytest.mark.parametrize("recipe,counts", [("grasp", (12, 4)), ("insertion", (26, 0))])
def test_contact_guide_prepares_templates_never_five_calls(asset, tmp_path, monkeypatch, recipe, counts):
    for name in ("analyze", "plan", "fit", "validate", "write", "inspect_usd"):
        monkeypatch.setattr(
            tuning, name, lambda *a, **k: pytest.fail("Contact collection must not execute calibration")
        )
    root = tmp_path / recipe
    first = start(asset=asset, goal="insertion transfer", directory=root, recipe=recipe)
    assert first["recipe_id"] == get_recipe(recipe)["id"]
    state = run(root, execute=True, simulation_profile="must-not-load.json", preview=lambda: pytest.fail("No preview"))
    assert state["state"] == "collection_spec_prepared"
    assert state["continuation"]["outcome"] == "awaiting_user"
    assert set(state["artifacts"]) == {"collection_spec"}
    assert not state["fit_allowed"] and not state["activation_allowed"]
    assert not state["real_robot_commands_sent"] and not state["hardware_execution_authorized"]
    assert "analysis_status" not in state
    spec = state["collection_spec"]
    assert (spec["core_trial_count"], spec["conditional_trial_count"]) == counts
    assert set(spec["scientific_calls"].values()) == {"not_run"}
    pack = Path(spec["workdir"])
    plan = verify_collection_bundle(pack)
    assert plan["numeric_motion_files"] == [] and plan["preview"]["status"] == "not_supported"
    assert len(list(pack.rglob("manifest.template.json"))) == sum(counts)
    for path in pack.rglob("manifest.template.json"):
        trial = json.loads(path.read_text())
        assert not trial["captured"] and trial["status"] == "not_collected"
        assert all(s["files"] == [] for s in trial["streams"].values())
    before = (pack / "bundle.json").read_bytes()
    assert advance(root, execute=True)["state"] == "collection_spec_prepared"
    assert (pack / "bundle.json").read_bytes() == before
    output = describe(state)
    assert "not executable motions" in output and "not implemented" in output
    assert "Lab checklist:" in output


def test_declarations_do_not_become_qualified_evidence_or_authority(asset, tmp_path):
    root = tmp_path / "session"
    start(asset=asset, goal="test", directory=root, recipe="grasp")
    setup = {"instrumentation": {"capabilities": {key: "available" for key in SIGNALS}}}
    provide(root, {"contact_setup": setup}, source="operator declaration")
    state = advance(root, execute=True)
    plan = verify_collection_bundle(state["collection_spec"]["workdir"])
    assert all(x["declared_availability"] == "available" and not x["qualified"] for x in plan["evidence_coverage"])
    assert not plan["hardware_execution_authorized"] and not plan["fit_allowed"]
    assert any(q["key"] == "contact_setup" for q in state["questions"])


def test_prior_inputs_and_evidence_preserved_new_revision_keeps_old_pack(asset, tmp_path):
    root = tmp_path / "session"
    start(asset=asset, goal="test", directory=root, recipe="insertion", evidence=tmp_path / "existing.json")
    first = advance(root)
    first_path = Path(first["collection_spec"]["workdir"])
    provide(root, {"contact_setup": {"geometry": {"object_id": "actual peg"}}}, source="operator")
    second = advance(root)
    assert first_path.is_dir() and second["collection_spec"]["workdir"] != str(first_path)
    plan = verify_collection_bundle(second["collection_spec"]["workdir"])
    assert plan["existing_evidence"] == str(tmp_path / "existing.json")
    assert plan["evidence_audit"] == "not_implemented_for_contact_recipes"
    assert second["history"][-1]["artifacts"]["collection_spec"]


def test_tampering_never_overwritten(asset, tmp_path):
    root = tmp_path / "session"
    start(asset=asset, goal="test", directory=root, recipe="grasp")
    first = advance(root)
    path = Path(first["collection_spec"]["workdir"]) / "setup.to_review.json"
    path.write_text('{"edited": true}')
    with pytest.raises(ValueError, match="Collection proposal changed"):
        advance(root)
    assert path.read_text() == '{"edited": true}'


@pytest.mark.parametrize(
    "patch",
    [
        {"instrumentation": {"capabilities": {"wrist_wrench": True}}},
        {"instrumentation": {"capabilities": {"made_up_signal": "available"}}},
        {"operator_review": "approved"},
        {"hardware_execution_authorized": True},
    ],
)
def test_invalid_setup_rejected_without_revision_change(asset, tmp_path, patch):
    root = tmp_path / "session"
    start(asset=asset, goal="test", directory=root, recipe="insertion")
    before = (root / "session.json").read_bytes()
    with pytest.raises(ValueError):
        provide(root, {"contact_setup": patch}, source="test")
    assert (root / "session.json").read_bytes() == before


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["execution"].update(fit_supported=True),
        lambda r: r["execution"].update(robot_execution_supported=True),
        lambda r: r["guided"].update(executor="arbitrary.code"),
        lambda r: r["trials"][0].update(code="../escape"),
        lambda r: r["trials"][0].update(repeats=True),
        lambda r: r["trials"].append(deepcopy(r["trials"][0])),
        lambda r: r["trials"][0].update(requires=["unknown"]),
        lambda r: r.update(trials=[t for t in r["trials"] if t["split"] != "heldout"]),
    ],
)
def test_recipe_validation(tmp_path, change):
    recipe = json.loads(CONTACT_RECIPES["grasp_contact@1"].read_text())
    change(recipe)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        load_collection_recipe(path)


def test_contact_checklist_escapes_text(asset, tmp_path):
    from newton_calibration.collection.contact import prepare_collection_bundle

    root = tmp_path / "session"
    state = start(asset=asset, goal="test", directory=root, recipe="grasp")
    definition = get_recipe("grasp")
    definition["title"] = '<img src=x onerror="bad()">'
    prepare_collection_bundle(tmp_path / "pack", state, definition)
    output = (tmp_path / "pack/LAB_CHECKLIST.html").read_text()
    assert "&lt;img" in output and "<img" not in output
    assert "Content-Security-Policy" in output
