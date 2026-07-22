from __future__ import annotations

from typing import Any

import numpy as np

from rice_tiller.postprocessing.outliers import filter_large_clusters
from rice_tiller.postprocessing.segments import class_id_by_name
from rice_tiller.types import HeightResult, SegmentationResult


def compute_plant_height(
    xyz_original: np.ndarray,
    xyz_normalized: np.ndarray,
    segmentation: SegmentationResult,
    config: dict[str, Any],
    *,
    seed: int = 42,
) -> HeightResult:
    pot_id = class_id_by_name(segmentation.class_names, "pot")
    plant_id = class_id_by_name(segmentation.class_names, "plant")
    pot_mask = segmentation.labels == pot_id
    plant_mask = segmentation.labels == plant_id
    n_pot = int(pot_mask.sum())
    n_plant = int(plant_mask.sum())
    minimum = int(config.get("minimum_class_points", 50))
    if n_pot < minimum or n_plant < minimum:
        return HeightResult(
            valid=False,
            n_pot=n_pot,
            n_plant=n_plant,
            warning=f"Need at least {minimum} Pot and Plant points",
        )

    pot_original = xyz_original[pot_mask]
    plant_original = xyz_original[plant_mask]
    pot_normalized = xyz_normalized[pot_mask]
    plant_normalized = xyz_normalized[plant_mask]

    dbscan_config = config.get("dbscan", {})
    pot_valid = filter_large_clusters(pot_normalized, dbscan_config, seed=seed)
    plant_valid = filter_large_clusters(plant_normalized, dbscan_config, seed=seed + 1)
    if int(pot_valid.sum()) >= minimum:
        pot_original = pot_original[pot_valid]
    else:
        pot_valid = np.ones(n_pot, dtype=bool)
    if int(plant_valid.sum()) >= minimum:
        plant_original = plant_original[plant_valid]
    else:
        plant_valid = np.ones(n_plant, dtype=bool)

    pot_percentile = float(config.get("pot_percentile", 99.0))
    plant_percentile = float(config.get("plant_percentile", 100.0))
    z_pot_top = float(np.percentile(pot_original[:, 2], pot_percentile))
    z_pot_bottom = float(np.percentile(pot_original[:, 2], 100.0 - pot_percentile))
    z_plant_top = float(np.percentile(plant_original[:, 2], plant_percentile))
    pot_height = z_pot_top - z_pot_bottom
    plant_height = z_plant_top - z_pot_top

    warning = None
    if pot_height <= 0:
        return HeightResult(
            valid=False,
            n_pot=n_pot,
            n_plant=n_plant,
            n_pot_valid=int(pot_valid.sum()),
            n_plant_valid=int(plant_valid.sum()),
            warning="Pot height is not positive",
            pot_valid_mask=pot_valid,
            plant_valid_mask=plant_valid,
        )

    if plant_height <= 0:
        strict_percentile = float(config.get("strict_pot_percentile", 95.0))
        strict_top = float(np.percentile(pot_original[:, 2], strict_percentile))
        strict_plant_height = z_plant_top - strict_top
        if strict_plant_height > 0:
            z_pot_top = strict_top
            plant_height = strict_plant_height
            warning = f"Used strict Pot percentile {strict_percentile:g} after sanity check"
        else:
            return HeightResult(
                valid=False,
                n_pot=n_pot,
                n_plant=n_plant,
                n_pot_valid=int(pot_valid.sum()),
                n_plant_valid=int(plant_valid.sum()),
                z_pot_top=z_pot_top,
                z_pot_bottom=z_pot_bottom,
                z_plant_top=z_plant_top,
                warning="Plant top is not above Pot top",
                pot_valid_mask=pot_valid,
                plant_valid_mask=plant_valid,
            )

    reference_height = float(config.get("pot_real_height_cm", 14.7))
    scale = reference_height / pot_height
    return HeightResult(
        valid=True,
        n_pot=n_pot,
        n_plant=n_plant,
        n_pot_valid=int(pot_valid.sum()),
        n_plant_valid=int(plant_valid.sum()),
        pot_height_pc=float(pot_height),
        plant_height_pc=float(plant_height),
        scale_cm_per_unit=float(scale),
        plant_height_cm=float(plant_height * scale),
        z_pot_top=float(z_pot_top),
        z_pot_bottom=float(z_pot_bottom),
        z_plant_top=float(z_plant_top),
        warning=warning,
        pot_valid_mask=pot_valid,
        plant_valid_mask=plant_valid,
    )

