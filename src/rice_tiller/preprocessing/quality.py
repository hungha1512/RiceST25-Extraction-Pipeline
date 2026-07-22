from __future__ import annotations

import numpy as np

from rice_tiller.types import PointCloud, QualityReport


def assess_input_quality(cloud: PointCloud, *, minimum_points: int = 4096) -> QualityReport:
    xyz = cloud.xyz
    report = QualityReport(valid=True)
    report.metrics.update(
        {
            "n_points": int(len(xyz)),
            "bounds_min": xyz.min(axis=0).tolist() if len(xyz) else None,
            "bounds_max": xyz.max(axis=0).tolist() if len(xyz) else None,
            "extent": np.ptp(xyz, axis=0).tolist() if len(xyz) else None,
        }
    )
    if len(xyz) < minimum_points:
        report.valid = False
        report.warnings.append(
            f"Point cloud has {len(xyz)} points; at least {minimum_points} are required"
        )
    if len(xyz) and np.ptp(xyz[:, 2]) <= 0:
        report.valid = False
        report.warnings.append("Point cloud has no positive Z extent")
    return report

