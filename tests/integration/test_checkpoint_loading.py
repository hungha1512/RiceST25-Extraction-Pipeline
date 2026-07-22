from pathlib import Path
import unittest

from rice_tiller.models.checkpoint import load_model_metadata


class CheckpointContractTest(unittest.TestCase):
    def test_default_metadata_contract(self) -> None:
        project = Path(__file__).resolve().parents[2]
        metadata = load_model_metadata(project / "weights" / "dgcnn_best" / "model.yaml")
        self.assertEqual(metadata["architecture"], "dgcnn")
        self.assertEqual(metadata["input"]["mode"], "xyz_h_n")
        self.assertEqual(metadata["input"]["channels"], 7)
        self.assertEqual(metadata["model"]["k"], 20)

    @unittest.skip("Requires a user-provided dg_best.pth checkpoint")
    def test_real_checkpoint_loads(self) -> None:
        pass
