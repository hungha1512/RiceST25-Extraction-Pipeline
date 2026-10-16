"""RiceST25-3D scans as training samples: random (train) or fixed (val/test) 4096-point subsets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.features import build_features, estimate_normals, normalize_xyz
from utils.pointcloud import read_ply


def pot_of(path) -> str:
    return Path(path).stem.split("_")[0]


def cached_normals(path, xyz, k: int, cache_dir: Path | None) -> np.ndarray:
    """Normals of the full cloud (file normals if present), cached as .npy next to the run outputs."""
    cache = cache_dir / f"{Path(path).stem}_n{k}.npy" if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache)
    normal = estimate_normals(xyz, k)
    if cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache, normal)
    return normal


class RiceDataset(Dataset):
    """Random (train) or fixed (val/test) subset of ``num_points`` points per labelled scan."""

    def __init__(self, files, mode="xyz_h_n", num_points=4096, augment=False, normal_k=20,
                 cache_dir=None, eval_seed=12345):
        self.files, self.mode, self.num_points, self.augment = list(files), mode, num_points, augment
        self.normal_k, self.cache_dir, self.eval_seed = normal_k, cache_dir, eval_seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = read_ply(self.files[idx])
        if data["label"] is None:
            raise ValueError(f"{self.files[idx]} has no Pot/Plant labels")
        n = len(data["xyz"])
        rng = np.random if self.augment else np.random.default_rng(self.eval_seed + idx)
        pick = rng.choice(n, self.num_points, replace=n < self.num_points)
        xyz, label = data["xyz"][pick], data["label"][pick]
        normal = None
        if self.mode == "xyz_h_n":
            full = data["normal"] if data["normal"] is not None else cached_normals(
                self.files[idx], data["xyz"], self.normal_k, self.cache_dir)
            normal = full[pick].copy()
        xyz = normalize_xyz(xyz)
        if self.augment:  # one rotation about z shared by coordinates and normals, then scale and jitter
            t = np.random.uniform(0, 2 * np.pi); c, s = np.cos(t), np.sin(t)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            xyz = (xyz @ rot.T) * np.float32(np.random.uniform(0.85, 1.15))
            xyz = xyz + np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
            normal = normal @ rot.T if normal is not None else None
        feat = build_features(xyz, normal, self.mode).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(feat)).T, torch.from_numpy(label).long()


def class_weights(files, num_classes=2) -> torch.Tensor:
    """Inverse-frequency weights from the training scans, clipped to [0.5, 10]."""
    counts = np.zeros(num_classes)
    for f in files:
        lab = read_ply(f)["label"]
        counts += np.bincount(lab, minlength=num_classes)
    counts = np.where(counts == 0, 1, counts)
    return torch.tensor(np.clip(counts.sum() / (num_classes * counts), 0.5, 10.0), dtype=torch.float32)
