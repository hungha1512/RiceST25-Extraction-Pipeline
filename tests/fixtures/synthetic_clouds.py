from __future__ import annotations

import numpy as np


def synthetic_pot_and_plant(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    pot = np.column_stack(
        [
            rng.uniform(-0.2, 0.2, n),
            rng.uniform(-0.2, 0.2, n),
            np.linspace(0.0, 1.0, n),
        ]
    )
    plant = np.column_stack(
        [
            rng.uniform(-0.15, 0.15, n),
            rng.uniform(-0.15, 0.15, n),
            np.linspace(1.1, 5.0, n),
        ]
    )
    xyz = np.vstack([pot, plant]).astype(np.float32)
    labels = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    return xyz, labels

