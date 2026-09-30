"""Measure circular bore sections in the user's binary STL; no asset modification."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def section(triangles, z):
    selected = triangles[(triangles[:, :, 2].min(1) < z) & (triangles[:, :, 2].max(1) > z)]
    segments = []
    for triangle in selected:
        points = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            if (a[2] - z) * (b[2] - z) < 0:
                points.append(a[:2] + (b[:2] - a[:2]) * (z - a[2]) / (b[2] - a[2]))
        if len(points) == 2:
            segments.append(points)
    return np.array(segments)


def circles(segments):
    graph = {}
    for a, b in segments:
        a, b = tuple(np.round(a, 4)), tuple(np.round(b, 4))
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    seen, result = set(), []
    for seed in graph:
        if seed in seen:
            continue
        pending, component = [seed], []
        while pending:
            point = pending.pop()
            if point in seen:
                continue
            seen.add(point)
            component.append(point)
            pending.extend(graph[point] - seen)
        xy = np.array(component)
        if len(xy) < 25 or np.ptp(xy, axis=0).min() < 20:
            continue
        fit = np.linalg.lstsq(np.column_stack([2 * xy, np.ones(len(xy))]), (xy * xy).sum(1), rcond=None)[0]
        radius = np.sqrt(fit[2] + (fit[:2] * fit[:2]).sum())
        error = np.abs(np.linalg.norm(xy - fit[:2], axis=1) - radius).max()
        if error < 0.03:
            result.append(
                {
                    "center_mm": fit[:2].tolist(),
                    "diameter_mm": float(2 * radius),
                    "max_fit_error_mm": float(error),
                    "vertices": len(xy),
                }
            )
    return sorted(result, key=lambda r: tuple(np.round(r["center_mm"], 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.fromfile(args.source, dtype=dtype, offset=84)["vertices"].astype(float)
    result = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "units": "millimeters, matching original USD conversion",
        "sections": [],
    }
    for z in [0.013, 5.013, 20.013, 35.013, 37.013, 38.013, 38.513, 39.013, 39.513, 39.913, 39.993]:
        lines = section(triangles, z)
        result["sections"].append({"z_mm": z, "circles": circles(lines)})
        if z == 39.913:
            canvas = Image.new("RGB", (1600, 1600), "white")
            draw = ImageDraw.Draw(canvas)
            for a, b in lines:
                draw.line(
                    [(800 + a[0] * 11, 800 - a[1] * 11), (800 + b[0] * 11, 800 - b[1] * 11)], fill="black", width=2
                )
            canvas.save(args.output / "fixture_top_section.png")
    (args.output / "hole_sections.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
