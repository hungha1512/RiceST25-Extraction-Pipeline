from __future__ import annotations

import numpy as np

from rice_tiller.exceptions import QualityError
from rice_tiller.types import NormalizationResult


def normalize_xyz(xyz: np.ndarray) -> NormalizationResult:
    points = np.asarray(xyz, dtype=np.float32)
    if len(points) == 0:
        raise QualityError("Cannot normalize an empty point cloud")
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise QualityError("Point cloud has zero or invalid spatial extent")
    return NormalizationResult(
        xyz=(centered / scale).astype(np.float32),
        centroid=centroid.astype(np.float32),
        scale=scale,
    )

