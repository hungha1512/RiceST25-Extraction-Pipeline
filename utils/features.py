"""Network input: normalised XYZ, relative height and surface normals (the 7D input)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

INPUT_DIM = {"xyz": 3, "xyz_h": 4, "xyz_h_n": 7}


def estimate_normals(xyz: np.ndarray, k: int = 20) -> np.ndarray:
    """PCA normals from the k nearest neighbours, flipped to face +z.

    Uses Open3D when installed (as in the research notebook); otherwise an equivalent
    NumPy/SciPy implementation, so Open3D is optional.
    """
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        pcd.orient_normals_to_align_with_direction([0, 0, 1])
        return np.nan_to_num(np.asarray(pcd.normals), nan=0.0).astype(np.float32)
    except ImportError:
        pass
    pts = xyz.astype(np.float64)
    _, nbr = cKDTree(pts).query(pts, k=min(k, len(pts)))
    local = pts[nbr] - pts[nbr].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local)
    normals = np.linalg.eigh(cov)[1][:, :, 0]          # eigenvector of the smallest eigenvalue
    normals[normals[:, 2] < 0] *= -1
    return normals.astype(np.float32)


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    """Centre on the centroid and scale by the largest distance from it (unit sphere)."""
    out = xyz - xyz.mean(0)
    radius = np.max(np.linalg.norm(out, axis=1))
    return out / radius if radius > 0 else out


def build_features(xyz_n: np.ndarray, normal: np.ndarray | None, mode: str = "xyz_h_n") -> np.ndarray:
    """(N, C) feature matrix from normalised coordinates; C = 3, 4 or 7."""
    if mode == "xyz":
        return xyz_n
    rel_h = (xyz_n[:, 2] - xyz_n[:, 2].min())[:, None]
    if mode == "xyz_h":
        return np.concatenate([xyz_n, rel_h], axis=1)
    if mode == "xyz_h_n":
        if normal is None:
            raise ValueError("mode 'xyz_h_n' needs surface normals")
        return np.concatenate([xyz_n, rel_h, normal], axis=1)
    raise ValueError(f"unknown input mode {mode!r}")
