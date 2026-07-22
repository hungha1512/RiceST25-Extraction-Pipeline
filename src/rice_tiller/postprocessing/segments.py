from __future__ import annotations

import numpy as np

from rice_tiller.exceptions import ConfigurationError


def class_id_by_name(class_names: dict[int, str], name: str) -> int:
    expected = name.strip().lower()
    matches = [class_id for class_id, value in class_names.items() if value.lower() == expected]
    if len(matches) != 1:
        raise ConfigurationError(f"Expected exactly one class named '{name}', got {class_names}")
    return int(matches[0])


def split_segments(
    xyz: np.ndarray,
    labels: np.ndarray,
    class_names: dict[int, str],
) -> dict[str, np.ndarray]:
    return {
        name: xyz[labels == class_id]
        for class_id, name in class_names.items()
    }

