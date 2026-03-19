from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from matchtile.calibration_store import calibration_profile_path, load_calibration, save_calibration
from matchtile.config import MatchTileConfig
from matchtile.models import Calibration, CellObservation, MatchGroup, Point, ReconstructionResult, Rect
from matchtile.vision import (
    CellFrameSample,
    TileFeature,
    _build_hidden_reference,
    _build_feature,
    _classify_sample_state,
    _draw_corner_magnifier,
    _extract_magnifier_crop,
    derive_center_anchored_calibration,
    _reveal_score,
    _refine_transition_states,
    build_manual_calibration,
    compare_features,
    enforce_required_group_size,
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
    def test_hybrid_descriptor_rescues_shape_similarity(self) -> None:
        left = np.full((28, 28, 3), (220, 60, 40), dtype=np.uint8)
        right = np.full((28, 28, 3), (210, 70, 50), dtype=np.uint8)
        cv2.circle(left, (14, 14), 8, (15, 15, 15), -1)
        cv2.circle(right, (13, 13), 8, (20, 20, 20), -1)
        cv2.rectangle(right, (0, 0), (4, 27), (210, 70, 50), -1)
        feature_left = _build_feature("left", left)
        feature_right = _build_feature("right", right)
        self.assertGreater(compare_features(feature_left, feature_right), 0.78)

    def test_transition_classifier_rejects_horizontal_squeeze_frames(self) -> None:
        hidden = np.full((28, 28, 3), 20, dtype=np.uint8)
        revealed = hidden.copy()
        revealed[:, :, :] = (210, 70, 40)
        cv2.circle(revealed, (14, 14), 6, (10, 10, 10), -1)
        squeezed = hidden.copy()
        squeezed[:, 10:18, :] = (210, 70, 40)
        cv2.circle(squeezed, (14, 14), 4, (10, 10, 10), -1)

        hidden_score, hidden_sharpness = _reveal_score(hidden)
        revealed_score, revealed_sharpness = _reveal_score(revealed)
        squeezed_score, squeezed_sharpness = _reveal_score(squeezed)
        hidden_sample = CellFrameSample(frame_index=0, crop=hidden, reveal_score=hidden_score, sharpness=hidden_sharpness)
        revealed_sample = CellFrameSample(frame_index=1, crop=revealed, reveal_score=revealed_score, sharpness=revealed_sharpness)
        squeezed_sample = CellFrameSample(frame_index=2, crop=squeezed, reveal_score=squeezed_score, sharpness=squeezed_sharpness)
        reference = hidden.copy()
        _classify_sample_state(hidden_sample, reference)
        _classify_sample_state(revealed_sample, reference)
        _classify_sample_state(squeezed_sample, reference)
        _refine_transition_states([hidden_sample, revealed_sample, squeezed_sample])

        self.assertEqual(hidden_sample.state, "hidden")
        self.assertEqual(revealed_sample.state, "fully-revealed")
        self.assertTrue(squeezed_sample.state.startswith("transition"))

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

    def test_extract_magnifier_crop_clamps_at_image_edges(self) -> None:
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        crop, focal = _extract_magnifier_crop(image, center_x=0, center_y=0, crop_size=11)
        self.assertEqual(crop.shape[:2], (11, 11))
        self.assertEqual(focal, (0, 0))
        crop2, focal2 = _extract_magnifier_crop(image, center_x=29, center_y=19, crop_size=11)
        self.assertEqual(crop2.shape[:2], (11, 11))
        self.assertEqual(focal2, (10, 10))

    def test_draw_corner_magnifier_renders_inset_on_canvas(self) -> None:
        image = np.full((120, 160, 3), 35, dtype=np.uint8)
        image[50:70, 70:90] = (220, 180, 60)
        canvas = image.copy()
        result = _draw_corner_magnifier(canvas, image, 80, 60, "Pick TL", crop_size=21, inset_size=120)
        self.assertEqual(result.shape, canvas.shape)
        self.assertGreater(float(np.mean(np.abs(result.astype(np.float32) - image.astype(np.float32)))), 1.0)

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

    def test_build_manual_calibration_re_subdivides_within_same_corners(self) -> None:
        base = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=40,
            corners=_empty_board_corners(),
            group_size=4,
        )
        denser = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=64,
            corners=base.board_corners,
            group_size=4,
        )
        self.assertEqual(denser.board_rect.x, base.board_rect.x)
        self.assertEqual(denser.board_rect.y, base.board_rect.y)
        self.assertEqual(denser.board_rect.width, base.board_rect.width)
        self.assertEqual(denser.board_rect.height, base.board_rect.height)
        self.assertLess(denser.pitch_x, base.pitch_x)
        self.assertLessEqual(denser.pitch_y, base.pitch_y)

    def test_center_anchored_derivation_preserves_pitch_and_shrinks_inward(self) -> None:
        base = build_manual_calibration(
            image_size=(2556, 1360),
            rows=25,
            cols=64,
            corners=_empty_board_corners(),
            group_size=4,
        )
        derived = derive_center_anchored_calibration(base, rows=25, cols=40, group_size=4)
        self.assertEqual(derived.rows, 25)
        self.assertEqual(derived.cols, 40)
        self.assertEqual(derived.group_size, 4)
        self.assertAlmostEqual(derived.pitch_x, base.anchor_pitch_x, places=2)
        self.assertAlmostEqual(derived.pitch_y, base.anchor_pitch_y, places=2)
        self.assertGreater(derived.board_rect.x, base.board_rect.x)
        self.assertLess(derived.board_rect.right, base.board_rect.right)
        self.assertGreaterEqual(derived.board_rect.y, base.board_rect.y)
        self.assertLessEqual(derived.board_rect.bottom, base.board_rect.bottom)

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
            group_size=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config.calibration_dir = tmp
            path = calibration_profile_path(config, "NULL//TRANSMIT.ERR", 2556, 1360)
            save_calibration(path, calibration)
            restored = load_calibration(path)
            self.assertEqual(restored.rows, 25)
            self.assertEqual(restored.cols, 64)
            self.assertEqual(restored.group_size, 4)
            self.assertEqual(len(restored.board_corners), 4)
            self.assertEqual(len(restored.anchor_corners), 4)
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
            self.assertTrue((session_dir / "candidate_debug.png").exists())
            self.assertEqual(Path(result.grid_fit_debug_path or ""), session_dir / "grid_fit_debug.png")
        self.assertEqual(Path(result.grid_composed_path or ""), session_dir / "grid_composed.png")
        self.assertEqual(Path(result.grid_debug_path or ""), session_dir / "grid_debug.png")
        self.assertEqual(Path(result.candidate_debug_path or ""), session_dir / "candidate_debug.png")
        self.assertGreater(len(result.observations), 50)
        self.assertGreaterEqual(len(result.groups), 2)

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

    def test_transition_regression_recovers_bottom_right_tile_and_triplets(self) -> None:
        session_dir = ROOT / "sessions" / "20260319-091130"
        session_frames = session_dir / "frames"
        calibration_path = session_dir / "calibration.json"
        if not session_frames.exists() or not calibration_path.exists():
            self.skipTest("Transition regression session is not present in this checkout.")
        calibration = load_calibration(calibration_path)
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80, max_group_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = reconstruct_from_images(
                sorted(session_frames.glob("*.png")),
                session_dir=out_dir,
                config=config,
                calibration=calibration,
            )
            self.assertTrue((out_dir / "candidate_debug.png").exists())
            self.assertIn("r20c51", result.observations)
            self.assertTrue(result.observations["r20c51"].crop_path)
            crop = load_image(Path(result.observations["r20c51"].crop_path))
            self.assertGreater(float(crop.std()), 20.0)
            self.assertTrue(any(group.group_size >= 3 for group in result.groups))
            self.assertTrue(all(not obs.state.startswith("transition") for obs in result.observations.values() if obs.crop_path))

    def test_session_regression_finds_known_four_match_and_click_order(self) -> None:
        session_dir = ROOT / "sessions" / "20260319-095451"
        session_frames = session_dir / "frames"
        calibration_path = session_dir / "calibration.json"
        if not session_frames.exists() or not calibration_path.exists():
            self.skipTest("4-match regression session is not present in this checkout.")
        calibration = load_calibration(calibration_path)
        config = MatchTileConfig(frame_stability_threshold=12.0, tile_match_threshold=0.72, group_confidence_threshold=0.80, max_group_size=4)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = reconstruct_from_images(
                sorted(session_frames.glob("*.png")),
                session_dir=out_dir,
                config=config,
                calibration=calibration,
            )
            target = {"r05c46", "r07c17", "r11c16", "r12c24"}
            matching = [group for group in result.groups if set(group.members) == target]
            self.assertEqual(len(matching), 1)
            group = matching[0]
            self.assertEqual(group.group_size, 4)
            self.assertEqual(group.click_order, ["r05c46", "r07c17", "r11c16", "r12c24"])
            self.assertTrue(group.click_order_positions)
            self.assertNotIn("r12c24", result.unresolved)
            self.assertNotIn("r05c46", result.unresolved)
            self.assertGreaterEqual(group.similarity_mean, group.similarity_min)

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
        self.assertTrue(all(group.click_order for group in groups))

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

    def test_enforce_required_group_size_moves_partial_groups_to_unresolved(self) -> None:
        calibration = Calibration(
            board_rect=Rect(0, 0, 100, 100),
            rows=1,
            cols=6,
            pitch_x=16.0,
            pitch_y=100.0,
            offset_x=0.0,
            offset_y=0.0,
            board_corners=[Point(0.0, 0.0), Point(100.0, 0.0), Point(100.0, 100.0), Point(0.0, 100.0)],
            client_width=100,
            client_height=100,
            rectified_width=100,
            rectified_height=100,
            centers=[],
        )
        observations = {
            "r00c00": CellObservation(0, 0, 0, 1.0, 1.0),
            "r00c01": CellObservation(0, 1, 0, 1.0, 1.0),
            "r00c02": CellObservation(0, 2, 0, 1.0, 1.0),
            "r00c03": CellObservation(0, 3, 0, 1.0, 1.0),
            "r00c04": CellObservation(0, 4, 0, 1.0, 1.0),
            "r00c05": CellObservation(0, 5, 0, 1.0, 1.0),
        }
        result = ReconstructionResult(
            calibration=calibration,
            observations=observations,
            groups=[
                MatchGroup(label="G001", members=["r00c00", "r00c01"], confidence=0.9, group_size=2),
                MatchGroup(
                    label="G002",
                    members=["r00c02", "r00c03", "r00c04", "r00c05"],
                    confidence=0.95,
                    group_size=4,
                ),
            ],
            unresolved=[],
            session_dir=".",
        )
        filtered = enforce_required_group_size(result, 4)
        self.assertEqual([group.label for group in filtered.groups], ["G002"])
        self.assertEqual(sorted(filtered.unresolved), ["r00c00", "r00c01"])

    def test_infer_match_groups_with_max_group_size_three_keeps_twos_and_threes(self) -> None:
        config = MatchTileConfig(tile_match_threshold=0.70, group_confidence_threshold=0.78, max_group_size=3)

        def feature(name: str, values: list[float]) -> TileFeature:
            vector = np.array(values, dtype=np.float32)
            vector = vector / np.linalg.norm(vector)
            return TileFeature(cell_id=name, descriptor=vector)

        features = {
            "a1": feature("a1", [1.0, 0.0, 0.0, 0.0]),
            "a2": feature("a2", [0.99, 0.01, 0.0, 0.0]),
            "b1": feature("b1", [0.0, 1.0, 0.0, 0.0]),
            "b2": feature("b2", [0.02, 0.98, 0.0, 0.0]),
            "b3": feature("b3", [0.01, 0.99, 0.0, 0.0]),
        }

        groups, unresolved = infer_match_groups(features, config)
        group_sizes = sorted(group.group_size for group in groups)
        self.assertEqual(group_sizes, [2, 3])
        self.assertFalse(unresolved)


if __name__ == "__main__":
    unittest.main()
