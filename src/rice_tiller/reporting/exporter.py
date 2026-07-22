from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rice_tiller.io.ply_writer import write_segmented_ply, write_xyz_ply
from rice_tiller.postprocessing.segments import class_id_by_name
from rice_tiller.reporting.schemas import RESULT_SCHEMA_VERSION
from rice_tiller.types import PipelineResult, PointCloud
from rice_tiller.visualization import plot_height, plot_segmentation, plot_tillers


def export_pipeline_result(
    result: PipelineResult,
    cloud: PointCloud,
    config: dict[str, Any],
) -> None:
    output_dir = result.output_dir
    point_dir = output_dir / "point_clouds"
    array_dir = output_dir / "arrays"
    figure_dir = output_dir / "figures"
    for directory in (output_dir, point_dir, array_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_result_summary(result, config)
    _write_json(output_dir / "preprocessing_report.json", result.quality.metrics)
    _write_measurements(output_dir / "measurements.csv", result)

    segmentation = result.segmentation
    if segmentation is None:
        return
    output_config = config.get("output", {})
    if output_config.get("save_segmented_cloud", True):
        write_segmented_ply(
            point_dir / "segmented.ply",
            cloud.xyz,
            segmentation.labels,
            segmentation.confidence,
            colors=cloud.colors,
        )

    pot_id = class_id_by_name(segmentation.class_names, "pot")
    plant_id = class_id_by_name(segmentation.class_names, "plant")
    pot_mask = segmentation.labels == pot_id
    plant_mask = segmentation.labels == plant_id
    if output_config.get("save_class_clouds", True):
        write_xyz_ply(
            point_dir / "pot.ply",
            cloud.xyz[pot_mask],
            colors=cloud.colors[pot_mask] if cloud.colors is not None else None,
        )
        write_xyz_ply(
            point_dir / "plant.ply",
            cloud.xyz[plant_mask],
            colors=cloud.colors[plant_mask] if cloud.colors is not None else None,
        )

    if result.tillers is not None and output_config.get("save_intermediate_arrays", True):
        if result.tillers.chm_raw is not None:
            np.save(array_dir / "chm_raw.npy", result.tillers.chm_raw)
        if result.tillers.chm_smoothed is not None:
            np.save(array_dir / "chm_smoothed.npy", result.tillers.chm_smoothed)
        _write_json(
            array_dir / "tiller_peaks.json",
            {
                "peak_coordinates": (
                    result.tillers.peak_coordinates.tolist()
                    if result.tillers.peak_coordinates is not None
                    else []
                ),
                "peak_xyz": (
                    result.tillers.peak_xyz.tolist() if result.tillers.peak_xyz is not None else []
                ),
            },
        )

    if output_config.get("save_visualizations", True):
        plot_segmentation(cloud.xyz, segmentation, figure_dir / "segmentation.png")
        if result.height is not None:
            plot_height(cloud.xyz, segmentation, result.height, figure_dir / "height_measurement.png")
        if result.tillers is not None:
            plot_tillers(
                cloud.xyz[plant_mask],
                result.tillers,
                figure_dir / "tiller_count.png",
            )


def write_result_summary(result: PipelineResult, config: dict[str, Any]) -> None:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    summary = result.summary_dict()
    summary["schema_version"] = RESULT_SCHEMA_VERSION
    summary["configuration"] = _public_config(config)
    _write_json(result.output_dir / "result.json", summary)


def _write_measurements(path: Path, result: PipelineResult) -> None:
    height = result.height
    tillers = result.tillers
    row = {
        "file": result.input_path.name,
        "status": result.status,
        "n_total": int(len(result.segmentation.labels)) if result.segmentation else 0,
        "n_pot": height.n_pot if height else None,
        "n_plant": height.n_plant if height else None,
        "n_pot_valid": height.n_pot_valid if height else None,
        "n_plant_valid": height.n_plant_valid if height else None,
        "pot_height_pc": height.pot_height_pc if height else None,
        "plant_height_pc": height.plant_height_pc if height else None,
        "scale_cm_per_unit": height.scale_cm_per_unit if height else None,
        "plant_height_cm": height.plant_height_cm if height else None,
        "n_tillers": tillers.n_tillers if tillers else None,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
