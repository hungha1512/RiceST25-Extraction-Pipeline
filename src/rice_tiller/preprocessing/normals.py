from __future__ import annotations

import numpy as np


def estimate_normals(xyz: np.ndarray, *, k: int = 20) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    if len(points) < 3:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(points), 1))
    k = max(3, min(int(k), len(points)))
    try:
        return _estimate_open3d(points, k)
    except ImportError:
        return _estimate_scipy(points, k)


def _estimate_open3d(points: np.ndarray, k: int) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    cloud.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    normals = np.asarray(cloud.normals, dtype=np.float32)
    return _clean_and_orient(normals)


def _estimate_scipy(points: np.ndarray, k: int, batch_size: int = 50000) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    normals = np.empty_like(points, dtype=np.float32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        _, neighbor_indices = tree.query(points[start:stop], k=k, workers=-1)
        neighbors = points[neighbor_indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("bnc,bnd->bcd", centered, centered) / max(k - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        normals[start:stop] = eigenvectors[:, :, 0].astype(np.float32)
    return _clean_and_orient(normals)


def _clean_and_orient(normals: np.ndarray) -> np.ndarray:
    output = np.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norm = np.linalg.norm(output, axis=1, keepdims=True)
    output = np.divide(output, norm, out=np.zeros_like(output), where=norm > 0)
    output[output[:, 2] < 0] *= -1
    invalid = np.linalg.norm(output, axis=1) == 0
    output[invalid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return output

