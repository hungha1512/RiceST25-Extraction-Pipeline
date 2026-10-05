import numpy as np
import pytest


def synthetic_scan(n_pot=6000, n_plant=3000, n_stems=3, seed=0, pot_h=0.147, plant_h=0.7):
    """A cylinder 'pot' (14.7 cm tall, base at z=0) and thin vertical 'stems' rising from its rim."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n_pot)
    pot = np.column_stack([0.05 * np.cos(ang), 0.05 * np.sin(ang), rng.uniform(0, pot_h, n_pot)])
    stems, per = [], n_plant // n_stems
    for i in range(n_stems):
        cx, cy = 0.03 * np.cos(2 * np.pi * i / n_stems), 0.03 * np.sin(2 * np.pi * i / n_stems)
        top = plant_h * (1 - 0.1 * i)
        stems.append(np.column_stack([cx + rng.normal(0, 0.004, per), cy + rng.normal(0, 0.004, per),
                                      rng.uniform(pot_h, top, per)]))
    plant = np.vstack(stems)
    xyz = np.vstack([pot, plant]).astype(np.float32)
    label = np.r_[np.zeros(len(pot)), np.ones(len(plant))].astype(np.int64)
    return xyz, label


@pytest.fixture
def scan():
    return synthetic_scan()
