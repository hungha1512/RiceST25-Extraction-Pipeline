from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rice_tiller.exceptions import CheckpointError, ConfigurationError
from rice_tiller.models.registry import create_model


def load_model_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise ConfigurationError(f"Model metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    _validate_metadata(metadata)
    metadata["_metadata_path"] = str(metadata_path.resolve())
    return metadata


def load_checkpoint_model(
    checkpoint_path: str | Path,
    metadata: dict[str, Any],
    *,
    device: str,
):
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("DGCNN inference requires PyTorch") from exc

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {checkpoint}. Copy dg_best.pth into the weights directory."
        )

    model_cfg = metadata["model"]
    input_cfg = metadata["input"]
    model = create_model(
        metadata["architecture"],
        num_classes=int(model_cfg["num_classes"]),
        in_channels=int(input_cfg["channels"]),
        k=int(model_cfg.get("k", 20)),
    )
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        state = _extract_state_dict(payload)
        state = {_remove_module_prefix(key): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise CheckpointError(f"Checkpoint does not match model metadata: {exc}") from exc
    model.to(device)
    model.eval()
    return model


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise CheckpointError("Checkpoint inspection requires PyTorch") from exc
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CheckpointError(f"Checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = _extract_state_dict(payload)
    return {
        "path": str(checkpoint.resolve()),
        "parameter_tensors": len(state),
        "parameter_count": int(sum(value.numel() for value in state.values())),
        "first_keys": list(state)[:10],
    }


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise CheckpointError("Unsupported checkpoint structure")


def _remove_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def _validate_metadata(metadata: dict[str, Any]) -> None:
    for key in ("architecture", "input", "model"):
        if key not in metadata:
            raise ConfigurationError(f"Model metadata is missing '{key}'")
    channels = int(metadata["input"].get("channels", 0))
    mode = metadata["input"].get("mode")
    expected = {"xyz": 3, "xyz_h": 4, "xyz_h_n": 7}.get(mode)
    if expected is None or channels != expected:
        raise ConfigurationError(
            f"Invalid model input contract: mode={mode!r}, channels={channels}"
        )

