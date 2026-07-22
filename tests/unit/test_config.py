from pathlib import Path
import unittest

from rice_tiller.config import load_config, resolve_project_path


class ConfigTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        project = Path(__file__).resolve().parents[2]
        config = load_config(project / "configs" / "default.yaml")
        self.assertEqual(config["model"]["architecture"], "dgcnn")
        self.assertEqual(config["tiller_count"]["raster"], {"rows": 35, "cols": 25})
        self.assertTrue(resolve_project_path(config, config["model"]["metadata"]).is_file())
