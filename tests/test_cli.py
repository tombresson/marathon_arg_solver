from __future__ import annotations

import unittest
from pathlib import Path

from matchtile.cli import _config_runtime_summary
from matchtile.config import ConfigLoadResult, MatchTileConfig


class CliTests(unittest.TestCase):
    def test_config_runtime_summary_reports_existing_config_file(self) -> None:
        config = MatchTileConfig(capture_fps=18, reveal_duration_s=10.0, max_group_size=4)
        summary = _config_runtime_summary(
            config,
            ConfigLoadResult(config=config, path=Path("matchtile.json"), created_default_file=False),
        )
        self.assertEqual(summary[0], "Config source: matchtile.json")
        self.assertIn("capture_fps=18", summary[1])
        self.assertIn("reveal_duration_s=10.00", summary[1])

    def test_config_runtime_summary_reports_created_defaults_file(self) -> None:
        config = MatchTileConfig()
        summary = _config_runtime_summary(
            config,
            ConfigLoadResult(config=config, path=Path("tmp-matchtile.json"), created_default_file=True),
        )
        self.assertEqual(summary[0], "Config source: defaults from code (saved to tmp-matchtile.json)")


if __name__ == "__main__":
    unittest.main()
