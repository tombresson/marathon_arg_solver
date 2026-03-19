from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from matchtile.calibration_store import calibration_profile_path, load_calibration, save_calibration
from matchtile.config import MatchTileConfig
from matchtile.models import Point
from matchtile.vision import (
    TileFeature,
    build_manual_calibration,
    infer_match_groups,
    load_image,
    reconstruct_from_images,
    warp_image_to_board,
)


ROOT = Path(__file__).resolve().parent.parent


def _empty_board_corners() -> list[Point]:
    return [
        Point(168.0, 366.0),
        Point(2197.0, 366.0),
        Point(2197.0, 1146.0),
        Point(168.0, 1146.0),
    ]


class VisionTests(unittest.TestCase):
    def test_build_manual_calibration_generates_projected_centers(self) -> None:
        calibration = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        self.assertEqual(calibration.rows, 25)
        self.assertEqual(calibration.cols, 64)
        self.assertEqual(len(calibration.centers), 1600)
        self.assertEqual(calibration.client_width, 2556)
        self.assertEqual(calibration.client_height, 1360)
        top_left = calibration.centers[0]
        bottom_right = calibration.centers[-1]
        self.assertLess(top_left.x, bottom_right.x)
        self.assertLess(top_left.y, bottom_right.y)

    def test_build_manual_calibration_changes_lattice_when_counts_change(self) -> None:
        smaller = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        larger = build_manual_calibration(
            image_size=(2556, 1360),
            rows=26,
            cols=65,
            corners=_empty_board_corners(),
        )
        self.assertNotEqual(smaller.cols, larger.cols)
        self.assertNotEqual(smaller.rows, larger.rows)
        self.assertLess(larger.pitch_x, smaller.pitch_x)
        self.assertLess(larger.pitch_y, smaller.pitch_y)

    def test_build_manual_calibration_clamps_minimum_pitch_geometry(self) -> None:
        calibration = build_manual_calibration(
            image_size=(2556, 1360),
            rows=1,
            cols=1,
            corners=_empty_board_corners(),
        )
        self.assertEqual(calibration.rows, 1)
        self.assertEqual(calibration.cols, 1)
        self.assertEqual(len(calibration.centers), 1)
        self.assertGreater(calibration.pitch_x, 0.0)
        self.assertGreater(calibration.pitch_y, 0.0)

    def test_locked_empty_boards_accept_manual_calibration(self) -> None:
        for name in ["empty-board-2.png", "empty-board-3.png"]:
            with self.subTest(name=name):
                image = load_image(ROOT / name)
                calibration = build_manual_calibration(
                    image_size=(image.shape[1], image.shape[0]),
                    rows=25,
                    cols=64,
                    corners=_empty_board_corners(),
                )
                board = warp_image_to_board(image, calibration)
                self.assertEqual(calibration.cols, 64)
                self.assertEqual(calibration.rows, 25)
                self.assertEqual(board.shape[1], calibration.rectified_width)
                self.assertEqual(board.shape[0], calibration.rectified_height)

    def test_calibration_store_round_trip(self) -> None:
        config = MatchTileConfig(calibration_dir="test-calibrations")
        calibration = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.calibration_dir = tmp
            path = calibration_profile_path(config, "NULL//TRANSMIT.ERR", 2556, 1360)
            save_calibration(path, calibration)
            restored = load_calibration(path)
            self.assertEqual(restored.rows, 25)
            self.assertEqual(restored.cols, 64)
            self.assertEqual(len(restored.board_corners), 4)
            self.assertEqual(restored.client_width, 2556)
            self.assertEqual(restored.client_height, 1360)

    def test_reconstruct_from_images_creates_groups(self) -> None:
        session_frames = ROOT / "sessions" / "20260319-024228" / "frames"
        if not session_frames.exists():
            self.skipTest("Captured regression session is not present in this checkout.")
        image = load_image(ROOT / "empty-board-2.png")
        calibration = build_manual_calibration(
            image_size=(image.shape[1], image.shape[0]),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        image_paths = sorted(session_frames.glob("*.png"))
        sample_paths = image_paths[:10] + image_paths[-12:]
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            result = reconstruct_from_images(
                sample_paths,
                session_dir=session_dir,
                config=config,
                calibration=calibration,
            )
            self.assertTrue((session_dir / "grid_fit_debug.png").exists())
            self.assertTrue((session_dir / "grid_composed.png").exists())
            self.assertTrue((session_dir / "grid_debug.png").exists())
            self.assertEqual(Path(result.grid_fit_debug_path or ""), session_dir / "grid_fit_debug.png")
            self.assertEqual(Path(result.grid_composed_path or ""), session_dir / "grid_composed.png")
            self.assertEqual(Path(result.grid_debug_path or ""), session_dir / "grid_debug.png")
        self.assertGreater(len(result.observations), 50)
        self.assertGreater(len(result.groups), 5)

    def test_session_regression_uses_manual_calibration_dimensions(self) -> None:
        session_frames = ROOT / "sessions" / "20260319-024228" / "frames"
        if not session_frames.exists():
            self.skipTest("Captured regression session is not present in this checkout.")
        image = load_image(ROOT / "empty-board-2.png")
        calibration = build_manual_calibration(
            image_size=(image.shape[1], image.shape[0]),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            image_paths = sorted(session_frames.glob("*.png"))[-12:]
            result = reconstruct_from_images(image_paths, session_dir=session_dir, config=config, calibration=calibration)
            self.assertEqual(result.calibration.cols, 64)
            self.assertEqual(result.calibration.rows, 25)
            self.assertTrue((session_dir / "grid_fit_debug.png").exists())

    def test_rendered_grid_matches_board_geometry(self) -> None:
        image = load_image(ROOT / "empty-board-2.png")
        calibration = build_manual_calibration(
            image_size=(image.shape[1], image.shape[0]),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
        )
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80)
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            result = reconstruct_from_images(
                [ROOT / "empty-board-2.png"],
                session_dir=session_dir,
                config=config,
                calibration=calibration,
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
