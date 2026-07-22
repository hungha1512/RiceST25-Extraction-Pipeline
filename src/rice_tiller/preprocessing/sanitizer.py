from __future__ import annotations

import numpy as np

from rice_tiller.types import PointCloud, QualityReport


def sanitize_point_cloud(
    cloud: PointCloud,
    *,
    remove_invalid: bool = True,
    remove_duplicates: bool = True,
    duplicate_tolerance: float = 1e-7,
) -> tuple[PointCloud, QualityReport]:
    xyz = cloud.xyz
    keep = np.ones(len(xyz), dtype=bool)
    invalid_count = 0
    if remove_invalid:
        finite = np.isfinite(xyz).all(axis=1)
        invalid_count = int((~finite).sum())
        keep &= finite

    indices = np.flatnonzero(keep)
    duplicate_count = 0
    if remove_duplicates and len(indices):
        values = xyz[indices]
        if duplicate_tolerance > 0:
            keys = np.rint(values / duplicate_tolerance).astype(np.int64)
        else:
            keys = values
        _, first = np.unique(keys, axis=0, return_index=True)
        first = np.sort(first)
        duplicate_count = int(len(indices) - len(first))
        indices = indices[first]

    sanitized = PointCloud(
        xyz=cloud.xyz[indices],
        colors=_take(cloud.colors, indices),
        intensity=_take(cloud.intensity, indices),
        source_indices=cloud.source_indices[indices],
        metadata={**cloud.metadata, "sanitized": True},
    )
    report = QualityReport(
        valid=len(sanitized.xyz) > 0,
        metrics={
            "raw_points": int(len(cloud.xyz)),
            "invalid_removed": invalid_count,
            "duplicates_removed": duplicate_count,
            "clean_points": int(len(sanitized.xyz)),
        },
    )
    if invalid_count:
        report.warnings.append(f"Removed {invalid_count} non-finite points")
    if duplicate_count:
        report.warnings.append(f"Removed {duplicate_count} duplicate points")
    return sanitized, report


def _take(value: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    return None if value is None else value[indices]

