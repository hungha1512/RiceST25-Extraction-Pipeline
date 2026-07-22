from __future__ import annotations

from typing import Any

from rice_tiller.exceptions import ConfigurationError


def create_model(name: str, **kwargs: Any):
    normalized = name.strip().lower()
    if normalized != "dgcnn":
        raise ConfigurationError(f"Unsupported model architecture: {name}")
    from rice_tiller.models.dgcnn import DGCNNSeg

    return DGCNNSeg(**kwargs)

