import numpy as np
from hole_variant import chamfer_penetration

SPEC = {"target_world_m": [0.115, 0.034, 0.79], "bore_radius_m": 0.012875, "chamfer_height_m": 0.002}


def test_chamfer_accepts_entry_but_not_same_offset_deep_in_bore():
    points = np.array([[0.115 + 0.014, 0.034, 0.7895], [0.115 + 0.014, 0.034, 0.78]])
    result = chamfer_penetration(points, SPEC)
    assert result[0] == 0
    assert result[1] > 0.001


def test_centered_peg_and_above_rim_are_free():
    points = np.array([[0.115 + 0.0125, 0.034, 0.755], [0.14, 0.034, 0.791]])
    assert np.all(chamfer_penetration(points, SPEC) == 0)


def test_other_real_holes_are_not_falsely_classified_as_solid():
    spec = {
        **SPEC,
        "all_holes": [SPEC, {"target_world_m": [0.15, 0, 0.79], "bore_radius_m": 0.01275, "chamfer_height_m": 0.0}],
    }
    assert chamfer_penetration(np.array([[0.15, 0, 0.785]]), spec)[0] == 0
