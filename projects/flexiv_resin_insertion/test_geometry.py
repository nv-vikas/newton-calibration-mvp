import numpy as np
import pytest
from geometry import block_penetration, cylinder_samples


def test_shallow_rim_contact_is_not_a_large_radial_penetration():
    points = np.array([[0.15 + 0.014, 0.0, 0.79 - 0.00001]])
    assert block_penetration(points)[0] == pytest.approx(0.00001)


def test_deep_wall_and_external_points():
    points = np.array([[0.15 + 0.014, 0.0, 0.77], [0.15, 0.0, 0.77], [0.3, 0, 0.77]])
    assert block_penetration(points).tolist() == pytest.approx([0.00125, 0.0, 0.0])


def test_aligned_seated_cylinder_is_clear_but_offset_cylinder_is_not():
    points = cylinder_samples().astype(np.float64)
    points[:, 2] = 0.827 - points[:, 2]
    points[:, 0] += 0.15
    assert block_penetration(points).max() == 0.0
    points[:, 0] += 0.0015
    assert block_penetration(points).max() > 0.001
