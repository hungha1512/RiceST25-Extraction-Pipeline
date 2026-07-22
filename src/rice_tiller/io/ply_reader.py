from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.exceptions import PointCloudIOError
from rice_tiller.types import PointCloud


def read_ply(path: str | Path) -> PointCloud:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise PointCloudIOError("PLY support requires the 'plyfile' package") from exc

    source = Path(path)
    try:
        vertex = PlyData.read(str(source))["vertex"]
    except Exception as exc:
        raise PointCloudIOError(f"Cannot read PLY file {source}: {exc}") from exc

    properties = {item.name for item in vertex.properties}
    missing = {"x", "y", "z"} - properties
    if missing:
        raise PointCloudIOError(f"PLY vertex is missing properties: {sorted(missing)}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    colors = _optional_colors(vertex, properties)
    intensity = None
    for name in ("intensity", "scalar_Intensity"):
        if name in properties:
            intensity = np.asarray(vertex[name], dtype=np.float32)
            break

    return PointCloud(
        xyz=xyz,
        colors=colors,
        intensity=intensity,
        metadata={
            "source_path": str(source.resolve()),
            "format": "ply",
            "vertex_properties": sorted(properties),
        },
    )


def _optional_colors(vertex: object, properties: set[str]) -> np.ndarray | None:
    triplets = [("red", "green", "blue"), ("r", "g", "b")]
    for names in triplets:
        if all(name in properties for name in names):
            return np.column_stack([vertex[name] for name in names]).astype(np.uint8)
    return None

