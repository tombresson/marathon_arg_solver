from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from matchtile.config import MatchTileConfig
import numpy as np

from matchtile.vision import TileFeature, calibrate_grid, crop_rect, detect_board_rect, detect_discord_activity_region, infer_match_groups, load_image, reconstruct_from_images


ROOT = Path(__file__).resolve().parent.parent


class VisionTests(unittest.TestCase):
    def test_detect_activity_region_contains_board(self) -> None:
        image = load_image(ROOT / "Screenshot 2026-03-19 011641.png")
        activity = detect_discord_activity_region(image)
        self.assertEqual(activity.x, 0)
        self.assertEqual(activity.y, 0)
        self.assertEqual(activity.width, image.shape[1])
        self.assertEqual(activity.height, image.shape[0])
        board = detect_board_rect(crop_rect(image, activity), restrict_left_bias=False)
        self.assertGreater(board.width, 900)
        self.assertGreater(board.height, 300)

    def test_calibrate_grid_estimates_large_board(self) -> None:
        image = load_image(ROOT / "Screenshot 2026-03-19 005448.png")
        board = detect_board_rect(image)
        calibration = calibrate_grid(crop_rect(image, board), board)
        self.assertGreaterEqual(calibration.rows, 20)
        self.assertGreaterEqual(calibration.cols, 40)
        self.assertGreater(len(calibration.centers), 800)

    def test_locked_empty_boards_calibrate_expected_dimensions(self) -> None:
        for name in ["empty-board-2.png", "empty-board-3.png"]:
            with self.subTest(name=name):
                image = load_image(ROOT / name)
                board = detect_board_rect(image)
                calibration = calibrate_grid(crop_rect(image, board), board)
                self.assertEqual(calibration.cols, 64)
                self.assertEqual(calibration.rows, 25)

    def test_reconstruct_from_images_creates_groups(self) -> None:
        session_frames = ROOT / "sessions" / "20260319-024228" / "frames"
        if not session_frames.exists():
            self.skipTest("Captured regression session is not present in this checkout.")
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        image_paths = sorted(session_frames.glob("*.png"))
        sample_paths = image_paths[:10] + image_paths[-12:]
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            result = reconstruct_from_images(
                sample_paths,
                session_dir=session_dir,
                config=config,
            )
            self.assertTrue((session_dir / "grid_fit_debug.png").exists())
            self.assertTrue((session_dir / "grid_composed.png").exists())
            self.assertTrue((session_dir / "grid_debug.png").exists())
            self.assertEqual(Path(result.grid_fit_debug_path or ""), session_dir / "grid_fit_debug.png")
            self.assertEqual(Path(result.grid_composed_path or ""), session_dir / "grid_composed.png")
            self.assertEqual(Path(result.grid_debug_path or ""), session_dir / "grid_debug.png")
        self.assertGreater(len(result.observations), 50)
        self.assertGreater(len(result.groups), 5)

    def test_session_regression_calibrates_expected_hidden_board(self) -> None:
        session_frames = ROOT / "sessions" / "20260319-024228" / "frames"
        if not session_frames.exists():
            self.skipTest("Captured regression session is not present in this checkout.")
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            image_paths = sorted(session_frames.glob("*.png"))[-12:]
            result = reconstruct_from_images(image_paths, session_dir=session_dir, config=config)
            self.assertEqual(result.calibration.cols, 64)
            self.assertEqual(result.calibration.rows, 25)
            self.assertTrue((session_dir / "grid_fit_debug.png").exists())

    def test_rendered_grid_matches_board_geometry(self) -> None:
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            result = reconstruct_from_images(
                [ROOT / "empty-board-2.png"],
                session_dir=session_dir,
                config=config,
            )
            composed = load_image(session_dir / "grid_composed.png")
            debug = load_image(session_dir / "grid_debug.png")
            expected_h = result.calibration.rows * int(round(result.calibration.pitch_y))
            expected_w = result.calibration.cols * int(round(result.calibration.pitch_x))
            self.assertEqual(composed.shape[0], expected_h)
            self.assertEqual(composed.shape[1], expected_w)
            self.assertEqual(debug.shape[0], expected_h)
            self.assertEqual(debug.shape[1], expected_w)

    def test_infer_match_groups_supports_mixed_sizes(self) -> None:
        config = MatchTileConfig(tile_match_threshold=0.70, group_confidence_threshold=0.78, max_group_size=4)

        def feature(name: str, values: list[float]) -> TileFeature:
            return TileFeature(cell_id=name, descriptor=np.array(values, dtype=np.float32))

        features = {
            "a1": feature("a1", [1.0, 0.0, 0.0, 0.0]),
            "a2": feature("a2", [0.98, 0.02, 0.0, 0.0]),
            "b1": feature("b1", [0.0, 1.0, 0.0, 0.0]),
            "b2": feature("b2", [0.02, 0.98, 0.0, 0.0]),
            "b3": feature("b3", [0.01, 0.99, 0.0, 0.0]),
            "c1": feature("c1", [0.0, 0.0, 1.0, 0.0]),
            "c2": feature("c2", [0.0, 0.0, 0.98, 0.02]),
            "c3": feature("c3", [0.0, 0.0, 0.99, 0.01]),
            "c4": feature("c4", [0.0, 0.0, 0.97, 0.03]),
        }
        for tile in features.values():
            tile.descriptor = tile.descriptor / np.linalg.norm(tile.descriptor)

        groups, unresolved = infer_match_groups(features, config)
        group_sizes = sorted(group.group_size for group in groups)
        self.assertEqual(group_sizes, [2, 3, 4])
        self.assertFalse(unresolved)

    def test_infer_match_groups_leaves_ambiguous_overlap_unresolved(self) -> None:
        config = MatchTileConfig(tile_match_threshold=0.78, group_confidence_threshold=0.82, max_group_size=4)

        def feature(name: str, values: list[float]) -> TileFeature:
            vector = np.array(values, dtype=np.float32)
            vector = vector / np.linalg.norm(vector)
            return TileFeature(cell_id=name, descriptor=vector)

        features = {
            "x1": feature("x1", [1.0, 0.0, 0.0]),
            "x2": feature("x2", [0.96, 0.28, 0.0]),
            "x3": feature("x3", [0.86, 0.51, 0.0]),
            "x4": feature("x4", [0.73, 0.68, 0.0]),
        }
        groups, unresolved = infer_match_groups(features, config)
        self.assertEqual(groups, [])
        self.assertEqual(sorted(unresolved), sorted(features))


if __name__ == "__main__":
    unittest.main()
