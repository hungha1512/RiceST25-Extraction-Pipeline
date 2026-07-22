import numpy as np
import unittest

from rice_tiller.preprocessing.sanitizer import sanitize_point_cloud
from rice_tiller.types import PointCloud


class SanitizerTest(unittest.TestCase):
    def test_removes_invalid_and_duplicate_points(self) -> None:
        xyz = np.array(
            [[0, 0, 0], [0, 0, 0], [1, 2, 3], [np.nan, 0, 1]],
            dtype=np.float32,
        )
        cloud, report = sanitize_point_cloud(PointCloud(xyz), duplicate_tolerance=1e-8)
        self.assertEqual(cloud.xyz.shape, (2, 3))
        self.assertEqual(report.metrics["invalid_removed"], 1)
        self.assertEqual(report.metrics["duplicates_removed"], 1)
