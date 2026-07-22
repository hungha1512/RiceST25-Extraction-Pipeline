from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rice_tiller.config import load_config, resolve_project_path
from rice_tiller.exceptions import QualityError
from rice_tiller.inference.segmenter import resolve_device, segment_full_cloud
from rice_tiller.io.reader import read_point_cloud
from rice_tiller.models.checkpoint import load_checkpoint_model, load_model_metadata
from rice_tiller.phenotypes.height import compute_plant_height
from rice_tiller.phenotypes.tillers import count_tillers_chm
from rice_tiller.postprocessing.segments import class_id_by_name
from rice_tiller.preprocessing.quality import assess_input_quality
from rice_tiller.preprocessing.sanitizer import sanitize_point_cloud
from rice_tiller.reporting.exporter import export_pipeline_result, write_result_summary
from rice_tiller.reporting.logger import configure_logger
from rice_tiller.types import PipelineResult


def run_pipeline(
    input_path: str | Path,
    *,
    config_path: str | Path,
    output_root: str | Path,
    scan_index: int | None = None,
) -> PipelineResult:
    source = Path(input_path).expanduser().resolve()
    config = load_config(config_path)
    if scan_index is not None:
        config["input"]["scan_index"] = int(scan_index)
    sample_name = source.stem
    output_dir = Path(output_root).expanduser().resolve() / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir / "pipeline.log")
    runtime: dict[str, float] = {}
    started = time.perf_counter()
    logger.info("Starting analysis for %s", source)

    mark = time.perf_counter()
    cloud = read_point_cloud(
        source,
        scan_index=int(config["input"].get("scan_index", 0)),
        apply_e57_pose=bool(config["input"].get("apply_e57_pose", True)),
    )
    runtime["input"] = time.perf_counter() - mark
    logger.info("Read %d points", cloud.n_points)

    mark = time.perf_counter()
    preprocessing = config.get("preprocessing", {})
    cloud, sanitize_report = sanitize_point_cloud(
        cloud,
        remove_invalid=bool(preprocessing.get("remove_invalid", True)),
        remove_duplicates=bool(preprocessing.get("remove_duplicates", True)),
        duplicate_tolerance=float(preprocessing.get("duplicate_tolerance", 1e-7)),
    )
    quality = assess_input_quality(
        cloud,
        minimum_points=int(preprocessing.get("minimum_points", 4096)),
    )
    quality.metrics = {**sanitize_report.metrics, **quality.metrics}
    quality.warnings = [*sanitize_report.warnings, *quality.warnings]
    runtime["preprocessing"] = time.perf_counter() - mark
    if not quality.valid:
        raise QualityError("; ".join(quality.warnings))
    logger.info("Sanitized cloud contains %d points", cloud.n_points)

    mark = time.perf_counter()
    metadata_path = resolve_project_path(config, config["model"]["metadata"])
    checkpoint_path = resolve_project_path(config, config["model"]["checkpoint"])
    metadata = load_model_metadata(metadata_path)
    configured_architecture = str(config["model"].get("architecture", "")).lower()
    if configured_architecture != str(metadata["architecture"]).lower():
        raise ValueError(
            f"Config architecture {configured_architecture!r} does not match model metadata "
            f"{metadata['architecture']!r}"
        )
    device = resolve_device(str(config["inference"].get("device", "auto")))
    config["inference"]["resolved_device"] = device
    model = load_checkpoint_model(checkpoint_path, metadata, device=device)
    runtime["model_loading"] = time.perf_counter() - mark
    logger.info("Loaded DGCNN checkpoint on %s", device)

    mark = time.perf_counter()
    segmentation, xyz_normalized = segment_full_cloud(
        cloud,
        model,
        metadata,
        config["inference"],
    )
    runtime["segmentation"] = time.perf_counter() - mark
    logger.info("Completed segmentation with %d voting passes", config["inference"]["voting_passes"])

    mark = time.perf_counter()
    height_config = {
        **config["height"],
        "minimum_class_points": int(config.get("postprocessing", {}).get("minimum_class_points", 50)),
    }
    height = (
        compute_plant_height(
            cloud.xyz,
            xyz_normalized,
            segmentation,
            height_config,
            seed=int(config["inference"].get("seed", 42)),
        )
        if bool(config["height"].get("enabled", True))
        else None
    )
    runtime["height"] = time.perf_counter() - mark

    mark = time.perf_counter()
    tillers = None
    if bool(config["tiller_count"].get("enabled", True)):
        plant_id = class_id_by_name(segmentation.class_names, "plant")
        plant_xyz = cloud.xyz[segmentation.labels == plant_id]
        tillers = count_tillers_chm(plant_xyz, config["tiller_count"])
    runtime["tiller_count"] = time.perf_counter() - mark
    if height is not None and height.warning:
        quality.warnings.append(f"Height: {height.warning}")
    if tillers is not None and tillers.warning:
        quality.warnings.append(f"Tiller count: {tillers.warning}")

    result = PipelineResult(
        sample_name=sample_name,
        input_path=source,
        output_dir=output_dir,
        status="success",
        quality=quality,
        segmentation=segmentation,
        height=height,
        tillers=tillers,
        runtime_seconds=runtime,
        model={
            "architecture": metadata["architecture"],
            "checkpoint": str(checkpoint_path),
            "metadata": str(metadata_path),
            "input_mode": metadata["input"]["mode"],
            "device": device,
        },
    )

    mark = time.perf_counter()
    export_pipeline_result(result, cloud, config)
    runtime["export"] = time.perf_counter() - mark
    runtime["total"] = time.perf_counter() - started
    # Refresh only result.json so it includes final export/total timings.
    write_result_summary(result, config)
    logger.info("Finished successfully in %.3f seconds", runtime["total"])
    return result
