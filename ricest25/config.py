"""Pipeline parameters. Defaults reproduce the manuscript; ``configs/default.yaml`` mirrors them."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "model": {"k": 20},   # graph neighbours; input channels / classes are read from the checkpoint
    "features": {"mode": "xyz_h_n", "normal_k": 20, "use_file_normals": True},
    "inference": {
        "device": "auto",
        "num_points": 4096,   # chunk size of the network
        "extra_passes": 3,    # random 4096-point groups added after the covering pass
        "batch_size": 8,      # chunks per forward pass (BatchNorm is in eval mode: no effect on output)
        "seed": 42,
    },
    "height": {
        "pot_height_cm": 14.7,
        "pot_percentile": 99.0,       # rim = z(99) of Pot; base = z(1) of Pot
        "plant_percentile": 100.0,    # apex = max z of Plant
        "strict_pot_percentile": 95.0,  # fallback rim if the plant apex falls below the rim
        "dbscan": {
            "enabled": True,
            "eps_m": 0.02,            # 2 cm, in the scanner's metric units
            "min_samples": 10,
            "min_cluster_size": 100,
            "map_radius_factor": 3.0,  # keep points within 3 * eps (6 cm) of a retained cluster
            "max_fit_points": 50000,
        },
        "min_class_points": 50,
    },
    "tillers": {
        "cell_size_m": 0.008,   # 8 mm CHM grid
        "sigma_cells": 1.0,     # Gaussian smoothing
        "window": 3,            # local-maximum window
        "min_height_ratio": 0.25,  # alpha
    },
    "evaluation": {"n_bootstrap": 10000, "bootstrap_seed": 0},
    "training": {
        "num_points": 4096, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 150, "patience": 20, "lopo_epochs": 60, "lopo_patience": 10,
        "seed": 42, "num_workers": 2,
    },
}


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    """Return the defaults, overridden key-by-key by an optional YAML file."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as handle:
            _merge(cfg, yaml.safe_load(handle) or {})
    return cfg
