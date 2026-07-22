from __future__ import annotations

from pathlib import Path

import numpy as np

from rice_tiller.exceptions import PointCloudIOError
from rice_tiller.types import PointCloud


def read_e57(path: str | Path, *, scan_index: int = 0, apply_pose: bool = True) -> PointCloud:
    try:
        import pye57
    except ImportError as exc:
        raise PointCloudIOError("E57 support requires the optional 'pye57' package") from exc

    source = Path(path)
    try:
        document = pye57.E57(str(source))
        if scan_index < 0 or scan_index >= document.scan_count:
            raise PointCloudIOError(
                f"scan_index={scan_index} is outside [0, {document.scan_count - 1}]"
            )
        data = document.read_scan(
            scan_index,
            intensity=True,
            colors=True,
            transform=apply_pose,
            ignore_missing_fields=True,
        )
    except PointCloudIOError:
        raise
    except Exception as exc:
        raise PointCloudIOError(f"Cannot read E57 file {source}: {exc}") from exc

    required = ("cartesianX", "cartesianY", "cartesianZ")
    if not all(name in data for name in required):
        raise PointCloudIOError("E57 scan does not contain Cartesian XYZ fields")

    xyz = np.column_stack([data[name] for name in required]).astype(np.float32)
    colors = None
    color_names = ("colorRed", "colorGreen", "colorBlue")
    if all(name in data for name in color_names):
        raw = np.column_stack([data[name] for name in color_names])
        if raw.size and np.nanmax(raw) <= 1.0:
            raw = raw * 255.0
        colors = np.clip(raw, 0, 255).astype(np.uint8)
    intensity = (
        np.asarray(data["intensity"], dtype=np.float32) if "intensity" in data else None
    )

    return PointCloud(
        xyz=xyz,
        colors=colors,
        intensity=intensity,
        metadata={
            "source_path": str(source.resolve()),
            "format": "e57",
            "scan_index": scan_index,
            "scan_count": int(document.scan_count),
            "pose_applied": bool(apply_pose),
        },
    )

