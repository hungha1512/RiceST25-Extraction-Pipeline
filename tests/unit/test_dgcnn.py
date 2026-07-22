import importlib.util
import unittest


class DGCNNTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
    def test_output_shape(self) -> None:
        import torch

        from rice_tiller.models.dgcnn import DGCNNSeg

        model = DGCNNSeg(num_classes=2, in_channels=7, k=4).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 7, 32))
        self.assertEqual(output.shape, (1, 32, 2))
