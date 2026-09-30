import pytest
from other_hole_env import assignments


def spec():
    return {
        "jobs": [
            {"starts": [{"source_heldout_index": k} for k in [14, 7, 2, 12]], "recorded_start_index": 0}
            for _ in range(4)
        ]
    }


def test_predeclared_recordings_and_matched_layout():
    layout = assignments(spec())
    assert len(layout) == 16
    assert [i for i, item in enumerate(layout) if item["record"]] == [0, 4, 8, 12]
    for job in range(4):
        assert [a["source_heldout_index"] for a in layout if a["job_index"] == job] == [14, 7, 2, 12]


def test_reject_missing_recorded_start():
    s = spec()
    s["jobs"][2]["recorded_start_index"] = 8
    with pytest.raises(ValueError):
        assignments(s)


def test_reject_unequal_banks():
    s = spec()
    s["jobs"][1]["starts"].pop()
    with pytest.raises(ValueError):
        assignments(s)
