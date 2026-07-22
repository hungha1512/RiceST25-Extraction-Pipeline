from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.exceptions import PointCloudIOError


def write_xyz_ply(
    path: str | Path,
    xyz: np.ndarray,
    *,
    colors: np.ndarray | None = None,
) -> Path:
    return _write_ply(path, xyz, colors=colors)


def write_segmented_ply(
    path: str | Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    *,
    colors: np.ndarray | None = None,
) -> Path:
    return _write_ply(
        path,
        xyz,
        colors=colors,
        labels=np.asarray(labels, dtype=np.uint8),
        confidence=np.asarray(confidence, dtype=np.float32),
    )


def _write_ply(
    path: str | Path,
    xyz: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> Path:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise PointCloudIOError("PLY support requires the 'plyfile' package") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    n_points = len(xyz)
    if xyz.shape != (n_points, 3):
        raise PointCloudIOError(f"xyz must have shape (N, 3), got {xyz.shape}")

    fields: list[tuple[str, str]] = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    if labels is not None:
        fields.append(("predicted_class", "u1"))
    if confidence is not None:
        fields.append(("confidence", "f4"))

    vertex = np.empty(n_points, dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    if colors is not None:
        rgb = np.asarray(colors)
        if rgb.shape != (n_points, 3):
            raise PointCloudIOError("colors must have shape (N, 3)")
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    if labels is not None:
        if labels.shape != (n_points,):
            raise PointCloudIOError("labels must have shape (N,)")
        vertex["predicted_class"] = labels
    if confidence is not None:
        if confidence.shape != (n_points,):
            raise PointCloudIOError("confidence must have shape (N,)")
        vertex["confidence"] = confidence

    try:
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(destination))
    except Exception as exc:
        raise PointCloudIOError(f"Cannot write PLY file {destination}: {exc}") from exc
    return destination

