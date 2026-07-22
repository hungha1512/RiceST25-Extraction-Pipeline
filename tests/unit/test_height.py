import numpy as np
import unittest

from rice_tiller.phenotypes.height import compute_plant_height
from rice_tiller.preprocessing.normalization import normalize_xyz
from rice_tiller.types import SegmentationResult
from tests.fixtures.synthetic_clouds import synthetic_pot_and_plant


class HeightTest(unittest.TestCase):
    def test_uses_pot_reference_scale(self) -> None:
        xyz, labels = synthetic_pot_and_plant()
        probabilities = np.eye(2, dtype=np.float32)[labels]
        segmentation = SegmentationResult(
            labels=labels,
            probabilities=probabilities,
            confidence=np.ones(len(labels), dtype=np.float32),
            vote_count=np.ones(len(labels), dtype=np.int32),
            class_names={0: "pot", 1: "plant"},
        )
        result = compute_plant_height(
            xyz,
            normalize_xyz(xyz).xyz,
            segmentation,
            {
                "minimum_class_points": 50,
                "pot_real_height_cm": 14.7,
                "pot_percentile": 100.0,
                "plant_percentile": 100.0,
                "dbscan": {"enabled": False},
            },
        )
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.pot_height_pc, 1.0, atol=1e-6)
        np.testing.assert_allclose(result.plant_height_cm, 58.8, atol=1e-5)
