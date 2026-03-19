from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from matchtile.config import MatchTileConfig


class ConfigTests(unittest.TestCase):
    def test_load_prefers_existing_json_values_over_code_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matchtile.json"
            path.write_text(
                json.dumps(
                    {
                        "capture_fps": 18,
                        "reveal_duration_s": 10.0,
                    }
                ),
                encoding="utf-8",
            )
            loaded = MatchTileConfig.load_with_result(path)
            self.assertFalse(loaded.created_default_file)
            self.assertEqual(loaded.path, path)
            self.assertEqual(loaded.config.reveal_duration_s, 10.0)
            self.assertEqual(loaded.config.capture_fps, 18)

    def test_load_uses_defaults_for_missing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matchtile.json"
            path.write_text(json.dumps({"capture_fps": 24}), encoding="utf-8")
            loaded = MatchTileConfig.load_with_result(path)
            self.assertEqual(loaded.config.capture_fps, 24)
            self.assertEqual(loaded.config.reveal_duration_s, MatchTileConfig().reveal_duration_s)
            self.assertEqual(loaded.config.click_delay_s, MatchTileConfig().click_delay_s)

    def test_load_creates_default_file_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matchtile.json"
            loaded = MatchTileConfig.load_with_result(path)
            self.assertTrue(loaded.created_default_file)
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["reveal_duration_s"], loaded.config.reveal_duration_s)


if __name__ == "__main__":
    unittest.main()
