from __future__ import annotations

from pathlib import Path

from rice_tiller.exceptions import PointCloudIOError
from rice_tiller.io.e57_reader import read_e57
from rice_tiller.io.ply_reader import read_ply
from rice_tiller.types import PointCloud


def read_point_cloud(
    path: str | Path,
    *,
    scan_index: int = 0,
    apply_e57_pose: bool = True,
) -> PointCloud:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PointCloudIOError(f"Point-cloud file not found: {source}")

    suffix = source.suffix.lower()
    if suffix == ".ply":
        return read_ply(source)
    if suffix == ".e57":
        return read_e57(source, scan_index=scan_index, apply_pose=apply_e57_pose)
    raise PointCloudIOError(f"Unsupported point-cloud format: {source.suffix}")

