"""Point-cloud input / output (PLY with optional per-point labels, optional E57)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def plant_mask_from_vertex(vertex, props: set[str]) -> np.ndarray | None:
    """Boolean Plant mask from a labelled RiceST25-3D PLY (``scalar_Plant`` non-NaN = Plant)."""
    if "scalar_Plant" in props:
        return ~np.isnan(np.asarray(vertex["scalar_Plant"], np.float32))
    if "label" in props:  # files written by this package
        return np.asarray(vertex["label"]) == 1
    if "scalar_Original_cloud_index" in props:  # CloudCompare "merge" export: the upper cloud is the plant
        idx, z = np.asarray(vertex["scalar_Original_cloud_index"]), np.asarray(vertex["z"])
        keys = np.unique(idx)
        if len(keys) != 2:
            raise ValueError("scalar_Original_cloud_index must hold exactly two values")
        return idx == max(keys, key=lambda k: z[idx == k].mean())
    return None


def read_ply(path: str | Path) -> dict:
    """Read a PLY. Returns ``xyz`` (N,3 float32), ``label`` (N, 0=Pot 1=Plant, or None) and ``normal``."""
    vertex = PlyData.read(str(path))["vertex"]
    props = {p.name for p in vertex.properties}
    if not {"x", "y", "z"} <= props:
        raise ValueError(f"{path}: vertex element has no x/y/z")
    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    mask = plant_mask_from_vertex(vertex, props)
    normal = None
    if {"nx", "ny", "nz"} <= props:
        normal = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(np.float32)
    return {"xyz": xyz, "label": None if mask is None else mask.astype(np.int64), "normal": normal}


def read_e57(path: str | Path, scan_index: int = 0) -> dict:
    """Read one scan of an E57 file (needs the optional ``pye57`` package)."""
    try:
        import pye57
    except ImportError as exc:
        raise ImportError("E57 input needs pye57:  pip install pye57") from exc
    doc = pye57.E57(str(path))
    data = doc.read_scan(scan_index, ignore_missing_fields=True, transform=True)
    xyz = np.column_stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]]).astype(np.float32)
    return {"xyz": xyz, "label": None, "normal": None}


def read_cloud(path: str | Path, scan_index: int = 0) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".e57":
        return read_e57(path, scan_index)
    raise ValueError(f"unsupported point-cloud format: {suffix}")


def write_ply(path: str | Path, xyz: np.ndarray, label: np.ndarray | None = None) -> None:
    """Write XYZ (+ optional per-point ``label``: 0=Pot, 1=Plant) as a binary PLY."""
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")] + ([("label", "u1")] if label is not None else [])
    vertex = np.empty(len(xyz), dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if label is not None:
        vertex["label"] = label
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))
