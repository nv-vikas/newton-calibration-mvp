"""Conservative geometry checks for a straight cylindrical bore with a linear chamfer.

This is a declared reference family, not a universal arbitrary-USD inspector.
Unsupported geometry is rejected rather than assigned invented precision.
"""

from itertools import pairwise

import numpy as np


def inspect_bore_triangles(triangles, *, center_xy, radius_m, height_m, chamfer_m, peg_radius_m):
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("Require finite triangulated collision geometry")
    xy = triangles[..., :2] - np.asarray(center_xy)
    radial = np.linalg.norm(xy, axis=-1)
    z = triangles[..., 2]
    expected = radius_m + np.maximum(z - (height_m - chamfer_m), 0)
    wall = np.all(np.abs(radial - expected) < 2e-7, axis=1) & (np.ptp(z, axis=1) > 1e-8)
    selected = triangles[wall]
    if not len(selected):
        raise ValueError("No matching straight bore/chamfer walls; unsupported or mis-scaled collider")
    levels = np.unique(selected[..., 2])
    if abs(levels[0]) > 1e-7 or abs(levels[-1] - height_m) > 1e-7:
        raise ValueError("Bore wall does not span the fixture height")
    minimum = float("inf")
    sections = []
    # Mesh segments are linear between vertex planes. Check every axial slab,
    # not a few hand-selected depths; report the faceted bore's inscribed radius.
    for low, high in pairwise(levels):
        for fraction in (1e-4, 0.5, 1 - 1e-4):
            plane = low + (high - low) * fraction
            candidates = selected[(selected[..., 2].min(1) < plane) & (selected[..., 2].max(1) > plane)]
            segments = []
            for tri in candidates:
                points = []
                for a, b in zip(tri, np.roll(tri, -1, axis=0)):
                    if (a[2] - plane) * (b[2] - plane) < 0:
                        points.append(a[:2] + (b[:2] - a[:2]) * (plane - a[2]) / (b[2] - a[2]))
                if len(points) == 2:
                    segments.append(points)
            if len(segments) < 32:
                raise ValueError("Insufficient/absent bore collision coverage in an axial slab")
            segments = np.array(segments) - center_xy
            graph = {}
            for a, b in segments:
                ka, kb = tuple(np.round(a, 10)), tuple(np.round(b, 10))
                graph.setdefault(ka, []).append(kb)
                graph.setdefault(kb, []).append(ka)
            if any(len(adjacent) != 2 for adjacent in graph.values()):
                raise ValueError("Open/nonmanifold bore boundary")
            visited, pending = set(), [next(iter(graph))]
            while pending:
                node = pending.pop()
                if node not in visited:
                    visited.add(node)
                    pending.extend(graph[node])
            if len(visited) != len(graph):
                raise ValueError("Multiple disconnected cross-section loops")
            a, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
            t = np.clip(-(a * delta).sum(1) / (delta * delta).sum(1), 0, 1)
            inner = np.linalg.norm(a + t[:, None] * delta, axis=1).min()
            # Account for the chamfer opening so the conservative straight bore radius is retained.
            inner -= max(plane - (height_m - chamfer_m), 0)
            minimum = min(minimum, float(inner))
            sections.append({"z_m": float(plane), "inscribed_bore_radius_m": float(inner), "segments": len(segments)})
    # Do not ignore an interior cap/plug simply because the side walls look right.
    other = triangles[~wall & (z.max(1) > 1e-7) & (z.min(1) < height_m - 1e-7)]
    if len(other):
        points = other[..., :2] - center_xy
        edges = np.roll(points, -1, axis=1) - points
        length2 = (edges * edges).sum(-1)
        t = np.clip(-(points * edges).sum(-1) / np.maximum(length2, 1e-30), 0, 1)
        closest = np.linalg.norm(points + t[..., None] * edges, axis=-1).min(1)
        # Origin inside the XY projection: all edge cross-products have one sign.
        cross = edges[..., 0] * -points[..., 1] - edges[..., 1] * -points[..., 0]
        inside = (
            ((cross >= 0).all(1) | (cross <= 0).all(1))
            & (np.ptp(points[..., 0], axis=1) > 1e-9)
            & (np.ptp(points[..., 1], axis=1) > 1e-9)
        )
        closest[inside] = 0
        if np.any(closest < minimum - 2e-7):
            raise ValueError("Additional collision surface obstructs the nominal bore")
    return {
        "minimum_radial_clearance_m": minimum - peg_radius_m,
        "inscribed_bore_radius_m": minimum,
        "geometry_error_bound_m": 2e-7,
        "coverage": "full_insertion_depth",
        "sections": sections,
        "method": "all axial wall slabs; conservative faceted inscribed radius for the supported straight-bore family",
        "limitation": "not an arbitrary solid-volume or manufacturing-tolerance certificate",
    }
