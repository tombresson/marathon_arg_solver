from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from matchtile.models import PhantomBoard, Point
from matchtile.phantom_board import (
    build_reconstruction_from_phantom_board,
    derive_board_calibration,
    load_phantom_board,
    parse_phantom_metadata,
    parse_phantom_board,
    phantom_board_hash,
    save_phantom_board,
)
from matchtile.vision import build_manual_calibration


def _corners() -> list[Point]:
    return [
        Point(100.0, 140.0),
        Point(740.0, 140.0),
        Point(740.0, 540.0),
        Point(100.0, 540.0),
    ]


def _sample_payload() -> dict:
    return {
        "width": 4,
        "height": 2,
        "pairSize": 2,
        "startedAt": "2026-03-19T15:28:06.000Z",
        "timestamp": "2026-03-19T15:28:06.938Z",
        "reportedAt": 1773934087419,
        "remainingTime": 992,
        "grid": [
            {"row": 0, "col": 0, "imgIdx": 7, "url": "https://invalid.local/7.png"},
            {"row": 0, "col": 1, "imgIdx": 9, "url": "https://invalid.local/9.png"},
            {"row": 0, "col": 2, "imgIdx": 11, "url": "https://invalid.local/11.png"},
            {"row": 0, "col": 3, "imgIdx": 9, "url": "https://invalid.local/9.png"},
            {"row": 1, "col": 0, "imgIdx": 11, "url": "https://invalid.local/11.png"},
            {"row": 1, "col": 1, "imgIdx": 7, "url": "https://invalid.local/7.png"},
            {"row": 1, "col": 2, "imgIdx": 5, "url": "https://invalid.local/5.png"},
            {"row": 1, "col": 3, "imgIdx": 5, "url": "https://invalid.local/5.png"},
        ],
    }


class PhantomBoardTests(unittest.TestCase):
    def test_parse_phantom_board_validates_counts(self) -> None:
        payload = _sample_payload()
        board = parse_phantom_board(payload)
        self.assertEqual(board.width, 4)
        self.assertEqual(board.height, 2)
        self.assertEqual(board.pairSize, 2)
        self.assertEqual(len(board.grid), 8)

    def test_parse_phantom_board_rejects_invalid_symbol_counts(self) -> None:
        payload = _sample_payload()
        payload["grid"][-1]["imgIdx"] = 7
        with self.assertRaises(RuntimeError):
            parse_phantom_board(payload)

    def test_parse_phantom_metadata_uses_dimensions_only(self) -> None:
        payload = _sample_payload()
        payload["grid"] = [{"row": 99, "col": 99, "imgIdx": 1, "url": "https://invalid.local/bad.png"}]
        metadata = parse_phantom_metadata(payload)
        self.assertEqual(metadata["width"], 4)
        self.assertEqual(metadata["height"], 2)
        self.assertEqual(metadata["pair_size"], 2)

    def test_board_hash_is_stable(self) -> None:
        board = PhantomBoard.from_dict(_sample_payload())
        self.assertEqual(phantom_board_hash(board), phantom_board_hash(board))

    def test_save_and_load_phantom_board_round_trip(self) -> None:
        board = parse_phantom_board(_sample_payload())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phantom_board.json"
            save_phantom_board(path, board)
            restored = load_phantom_board(path)
            self.assertEqual(restored.to_dict(), board.to_dict())

    def test_derive_board_calibration_respects_runtime_dimensions(self) -> None:
        base_calibration = build_manual_calibration(
            image_size=(900, 700),
            rows=1,
            cols=1,
            corners=_corners(),
        )
        derived = derive_board_calibration(base_calibration, rows=2, cols=4)
        self.assertEqual(derived.rows, 2)
        self.assertEqual(derived.cols, 4)
        self.assertEqual(len(derived.centers), 8)
        self.assertEqual(derived.client_width, base_calibration.client_width)
        self.assertEqual(derived.client_height, base_calibration.client_height)

    def test_build_reconstruction_from_phantom_board_creates_exact_groups(self) -> None:
        base_calibration = build_manual_calibration(
            image_size=(900, 700),
            rows=1,
            cols=1,
            corners=_corners(),
        )
        board = parse_phantom_board(_sample_payload())
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            result = build_reconstruction_from_phantom_board(
                board,
                base_calibration,
                session_dir=session_dir,
                board_source={"mode": "test"},
            )
            self.assertEqual(result.calibration.rows, 2)
            self.assertEqual(result.calibration.cols, 4)
            self.assertEqual(len(result.calibration.centers), 8)
            self.assertEqual(len(result.groups), 4)
            self.assertEqual(len(result.observations), 8)
            self.assertFalse(result.unresolved)
            self.assertTrue((session_dir / "grid_fit_debug.png").exists())
            self.assertTrue((session_dir / "grid_composed.png").exists())
            self.assertTrue((session_dir / "grid_debug.png").exists())
            ordered_groups = [(group.members, group.click_order_positions) for group in result.groups]
            self.assertEqual(ordered_groups[0][0], ["r00c00", "r01c01"])
            self.assertEqual(ordered_groups[1][0], ["r00c01", "r00c03"])
            self.assertEqual(ordered_groups[0][1][0], {"row": 0, "col": 0})
            self.assertEqual(result.board_source, {"mode": "test"})


if __name__ == "__main__":
    unittest.main()
