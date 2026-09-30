import numpy as np
import pytest

from newton_calibration.qualification.geometry import inspect_bore_triangles


def walls(n=128):
    triangles = []
    for low, high, r0, r1 in [(0, 0.038, 0.012525, 0.012525), (0.038, 0.04, 0.012525, 0.014525)]:
        for i in range(n):
            a, b = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
            p = [r0 * np.cos(a), r0 * np.sin(a), low]
            q = [r0 * np.cos(b), r0 * np.sin(b), low]
            s = [r1 * np.cos(a), r1 * np.sin(a), high]
            t = [r1 * np.cos(b), r1 * np.sin(b), high]
            triangles.extend([[p, q, s], [q, t, s]])
    return np.array(triangles)


def inspect(triangles):
    return inspect_bore_triangles(
        triangles,
        center_xy=np.array([0.0, 0.0]),
        radius_m=0.012525,
        height_m=0.04,
        chamfer_m=0.002,
        peg_radius_m=0.0125,
    )


def test_clearance_accounts_for_facets_not_just_nominal_radius():
    result = inspect(walls())
    assert 20e-6 < result["minimum_radial_clearance_m"] < 25e-6
    assert len(result["sections"]) == 6


def test_missing_wall_fails():
    with pytest.raises(ValueError, match="Open/nonmanifold"):
        inspect(walls()[2:])


def test_plug_is_not_hidden_by_valid_side_walls():
    cap = [[-0.01, -0.01, 0.02], [0.01, -0.01, 0.02], [0, 0.01, 0.02]]
    with pytest.raises(ValueError, match="obstructs"):
        inspect(np.concatenate([walls(), [cap]]))


def test_wrong_scale_fails():
    with pytest.raises(ValueError):
        inspect(walls() * 1000)
