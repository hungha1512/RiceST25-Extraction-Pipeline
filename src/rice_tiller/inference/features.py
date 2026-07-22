from __future__ import annotations

import numpy as np

from rice_tiller.exceptions import ConfigurationError
from rice_tiller.preprocessing.normals import estimate_normals
from rice_tiller.types import NormalizationResult


def build_features(
    normalization: NormalizationResult,
    original_xyz: np.ndarray,
    *,
    mode: str,
    normal_k: int = 20,
) -> np.ndarray:
    xyz = normalization.xyz
    if mode == "xyz":
        return xyz.astype(np.float32)

    relative_height = (xyz[:, 2] - xyz[:, 2].min()).reshape(-1, 1).astype(np.float32)
    if mode == "xyz_h":
        return np.concatenate([xyz, relative_height], axis=1).astype(np.float32)

    if mode == "xyz_h_n":
        normals = estimate_normals(original_xyz, k=normal_k)
        return np.concatenate([xyz, relative_height, normals], axis=1).astype(np.float32)

    raise ConfigurationError(f"Unsupported feature mode: {mode}")

