from __future__ import annotations

from typing import Any

import numpy as np

from rice_tiller.inference.features import build_features
from rice_tiller.inference.voting import full_cloud_voting
from rice_tiller.preprocessing.normalization import normalize_xyz
from rice_tiller.types import PointCloud, SegmentationResult


def segment_full_cloud(
    cloud: PointCloud,
    model,
    metadata: dict[str, Any],
    inference_config: dict[str, Any],
) -> tuple[SegmentationResult, np.ndarray]:
    normalization = normalize_xyz(cloud.xyz)
    input_config = metadata["input"]
    model_config = metadata["model"]
    requested_chunk_size = int(inference_config.get("chunk_size", input_config["num_points"]))
    trained_num_points = int(input_config["num_points"])
    if requested_chunk_size != trained_num_points:
        raise ValueError(
            f"inference.chunk_size={requested_chunk_size} does not match the checkpoint "
            f"contract num_points={trained_num_points}"
        )
    features = build_features(
        normalization,
        cloud.xyz,
        mode=input_config["mode"],
        normal_k=int(input_config.get("normal_k", 20)),
    )
    expected_channels = int(input_config["channels"])
    if features.shape[1] != expected_channels:
        raise ValueError(
            f"Feature builder produced {features.shape[1]} channels; expected {expected_channels}"
        )

    probabilities, counts = full_cloud_voting(
        model,
        features,
        num_classes=int(model_config["num_classes"]),
        chunk_size=requested_chunk_size,
        batch_size=int(inference_config.get("batch_size", 1)),
        passes=int(inference_config.get("voting_passes", 3)),
        seed=int(inference_config.get("seed", 42)),
        device=str(inference_config["resolved_device"]),
    )
    labels = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1).astype(np.float32)
    class_names = {int(key): str(value) for key, value in model_config["class_names"].items()}
    return (
        SegmentationResult(
            labels=labels,
            probabilities=probabilities,
            confidence=confidence,
            vote_count=counts,
            class_names=class_names,
        ),
        normalization.xyz,
    )


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("DGCNN inference requires PyTorch") from exc

    requested = requested.lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
