import numpy as np
import unittest

from rice_tiller.preprocessing.normalization import normalize_xyz


class NormalizationTest(unittest.TestCase):
    def test_centers_and_scales_full_cloud(self) -> None:
        xyz = np.array([[0, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=np.float32)
        result = normalize_xyz(xyz)
        np.testing.assert_allclose(result.xyz.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(result.xyz, axis=1).max(), 1.0, atol=1e-6)
