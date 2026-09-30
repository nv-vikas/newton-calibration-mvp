"""Advice is a projection of recorded readiness, never permission to execute."""

from copy import deepcopy

import pytest

from newton_calibration.guided.actions import build_action_plan
from newton_calibration.guided.cli import describe


def blocked_session():
    return {
        "session_id": "synthetic-review",
        "revision": 3,
        "state": "needs_information",
        "analysis_status": "completed",
        "fit_allowed": False,
        "inputs": {"evidence": {"root": "existing-logs"}},
        "questions": [
            {"key": key}
            for key in (
                "controller_rate",
                "controller",
                "environment",
                "confirm.mapping",
                "confirm.controller",
                "confirm.tool",
                "confirm.bounds",
                "readiness",
            )
        ],
        "evidence_needs": {"heldout_needed": False, "parameters": [{"disposition": "fit"}]},
    }


def test_setup_gap_has_four_owned_actions_not_blanket_recollection():
    session = blocked_session()
    before = deepcopy(session)
    plan = build_action_plan(session)
    assert session == before  # pure advice; does not confirm inputs or change readiness
    assert [a["id"] for a in plan["items"]] == ["qualify_replay", "prepare_profile", "confirm_setup", "recheck"]
    assert plan["items"][2]["depends_on"] == ["prepare_profile"]
    assert plan["items"][-1]["depends_on"] == ["qualify_replay", "prepare_profile", "confirm_setup"]
    assert all(a["owner"] and a["done_when"] and a["status"] == "proposed" for a in plan["items"])
    assert plan["data_collection"]["status"] == "deferred"
    assert "No new robot data requested yet" in plan["data_collection"]["message"]
    assert "Do not change" in plan["items"][0]["done_when"]
    output = describe({**session, "action_plan": plan})
    assert "What to do next:" in output and "Done when:" in output
    assert "Toolkit engineering / agent" in output and "Robot engineer" in output


@pytest.mark.parametrize(
    "data_present,questions,expected",
    [
        (False, [], "needed"),
        (False, ["controller"], "deferred"),
        (True, [], "not_requested"),
        (True, ["evidence"], "deferred"),
    ],
)
def test_collection_decision_respects_existing_evidence(data_present, questions, expected):
    session = blocked_session()
    session["questions"] = [{"key": q} for q in questions]
    session["inputs"]["evidence"] = {"existing": True} if data_present else None
    plan = build_action_plan(session)
    assert plan["data_collection"]["status"] == expected
    assert any(a["id"] == "prepare_collection" for a in plan["items"]) == (not data_present)


def test_missing_holdout_requests_targeted_collection():
    session = blocked_session()
    session["questions"] = [{"key": "collection"}]
    session["evidence_needs"]["heldout_needed"] = True
    plan = build_action_plan(session)
    assert plan["data_collection"]["status"] == "needed"
    assert "remaining evidence gaps" in plan["items"][0]["action"]
    assert "operator" in plan["items"][0]["done_when"].lower()


@pytest.mark.parametrize(
    "state,allowed,error,action",
    [
        ("choose_recipe", False, None, "choose_recipe"),
        ("ready_to_fit", True, None, "authorize_fit"),
        ("fitting", True, None, "monitor"),
        ("completed", True, None, "review_result"),
        ("needs_attention", False, {"step": "fit"}, "recover"),
    ],
)
def test_progress_states_do_not_recommend_a_blind_rerun(state, allowed, error, action):
    session = {**blocked_session(), "state": state, "fit_allowed": allowed, "error": error}
    assert build_action_plan(session)["items"][0]["id"] == action


def test_uninterpretable_data_and_unsupported_controller_are_not_fit_ready():
    session = blocked_session()
    session["state"] = "needs_evidence_description"
    session["questions"] = [{"key": "evidence"}, {"key": "controller_support"}]
    ids = [a["id"] for a in build_action_plan(session)["items"]]
    assert ids == ["qualify_replay", "interpret_evidence", "recheck"]
    assert session["fit_allowed"] is False
