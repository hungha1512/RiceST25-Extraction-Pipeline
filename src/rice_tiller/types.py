from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PointCloud:
    xyz: np.ndarray
    colors: np.ndarray | None = None
    intensity: np.ndarray | None = None
    source_indices: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")
        if self.source_indices is None:
            self.source_indices = np.arange(len(self.xyz), dtype=np.int64)

    @property
    def n_points(self) -> int:
        return int(len(self.xyz))


@dataclass
class NormalizationResult:
    xyz: np.ndarray
    centroid: np.ndarray
    scale: float


@dataclass
class QualityReport:
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    labels: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    vote_count: np.ndarray
    class_names: dict[int, str]


@dataclass
class HeightResult:
    valid: bool
    n_pot: int = 0
    n_plant: int = 0
    n_pot_valid: int = 0
    n_plant_valid: int = 0
    pot_height_pc: float | None = None
    plant_height_pc: float | None = None
    scale_cm_per_unit: float | None = None
    plant_height_cm: float | None = None
    z_pot_top: float | None = None
    z_pot_bottom: float | None = None
    z_plant_top: float | None = None
    warning: str | None = None
    pot_valid_mask: np.ndarray | None = field(default=None, repr=False)
    plant_valid_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class TillerResult:
    valid: bool
    n_tillers: int = 0
    raster_shape: tuple[int, int] = (35, 25)
    gaussian_sigma_px: float = 1.0
    min_distance_px: int = 3
    peak_coordinates: np.ndarray | None = field(default=None, repr=False)
    peak_xyz: np.ndarray | None = field(default=None, repr=False)
    chm_raw: np.ndarray | None = field(default=None, repr=False)
    chm_smoothed: np.ndarray | None = field(default=None, repr=False)
    valid_mask: np.ndarray | None = field(default=None, repr=False)
    warning: str | None = None


@dataclass
class PipelineResult:
    sample_name: str
    input_path: Path
    output_dir: Path
    status: str
    quality: QualityReport
    segmentation: SegmentationResult | None = field(default=None, repr=False)
    height: HeightResult | None = None
    tillers: TillerResult | None = None
    runtime_seconds: dict[str, float] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def summary_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_name": self.sample_name,
            "input_path": str(self.input_path),
            "status": self.status,
            "quality": asdict(self.quality),
            "runtime_seconds": self.runtime_seconds,
            "model": self.model,
            "error": self.error,
        }
        if self.segmentation is not None:
            labels = self.segmentation.labels
            result["segmentation"] = {
                "n_points": int(len(labels)),
                "class_counts": {
                    self.segmentation.class_names.get(int(c), str(c)): int((labels == c).sum())
                    for c in np.unique(labels)
                },
                "mean_confidence": float(self.segmentation.confidence.mean()),
                "minimum_vote_count": int(self.segmentation.vote_count.min()),
            }
        if self.height is not None:
            result["height"] = _without_arrays(asdict(self.height))
        if self.tillers is not None:
            tiller = _without_arrays(asdict(self.tillers))
            if self.tillers.peak_coordinates is not None:
                tiller["peak_coordinates"] = self.tillers.peak_coordinates.tolist()
            if self.tillers.peak_xyz is not None:
                tiller["peak_xyz"] = self.tillers.peak_xyz.tolist()
            result["tiller_count"] = tiller
        return result


def _without_arrays(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not isinstance(item, np.ndarray)}

