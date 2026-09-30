"""Independent geometric screens for this known cylindrical peg and bored block."""

import numpy as np


def cylinder_samples():
    angles = np.arange(64) * (2 * np.pi / 64)
    z = np.linspace(0, 0.075, 41)
    side = np.stack(np.meshgrid(angles, z, indexing="ij"), axis=-1).reshape(-1, 2)
    points = np.column_stack([0.0125 * np.cos(side[:, 0]), 0.0125 * np.sin(side[:, 0]), side[:, 1]])
    caps = []
    for height in [0.0, 0.075]:
        for radius in np.linspace(0, 0.0125, 9):
            caps.extend(np.column_stack([radius * np.cos(angles), radius * np.sin(angles), angles * 0 + height]))
    return np.concatenate([points, caps]).astype(np.float32)


def block_penetration(points, xp=np):
    """Per-point penetration into the solid block, not mere rim misalignment.

    Points are environment-local. The nearest free surface can be the bore,
    top/bottom, or outer block boundary. A tip 10 um below the rim is NOT a
    1 mm penetration merely because its horizontal alignment error is 1 mm.
    This sampled analytic screen complements, not replaces, collision contacts.
    """
    x, y, z = points[..., 0] - 0.15, points[..., 1], points[..., 2]
    radial = xp.sqrt(x * x + y * y)
    distance = xp.minimum(radial - 0.01275, xp.minimum(0.79 - z, z - 0.75))
    distance = xp.minimum(distance, xp.minimum(0.0635 - xp.abs(x), 0.0635 - xp.abs(y)))
    return xp.maximum(distance, xp.zeros_like(distance))
