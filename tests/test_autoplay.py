from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from pynput.mouse import Button

from matchtile.autoplay import _execute_click, _jittered_delay, auto_click_groups
from matchtile.models import Calibration, CellCenter, CellObservation, MatchGroup, ReconstructionResult, Rect


class DummyController:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self._position: tuple[int, int] | None = None

    @property
    def position(self) -> tuple[int, int] | None:
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._position = value
        self.events.append(("move", value))

    def press(self, button: Button) -> None:
        self.events.append(("press", button))

    def release(self, button: Button) -> None:
        self.events.append(("release", button))


class DummyStopToken:
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.checkpoints = 0

    def sleep(self, duration_s: float, step_s: float = 0.02) -> None:
        del step_s
        self.sleeps.append(duration_s)

    def checkpoint(self) -> None:
        self.checkpoints += 1


class DummyMSSContext:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def __enter__(self) -> "DummyMSSContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def grab(self, monitor: dict[str, int]) -> np.ndarray:
        del monitor
        return self.frame


def _cell_id(row: int, col: int) -> str:
    return f"r{row:02d}c{col:02d}"


def _build_result(session_dir: Path) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    calibration = Calibration(
        board_rect=Rect(0, 0, 320, 120),
        rows=1,
        cols=4,
        pitch_x=80.0,
        pitch_y=80.0,
        offset_x=0.0,
        offset_y=0.0,
        group_size=4,
        client_width=320,
        client_height=120,
        rectified_width=320,
        rectified_height=120,
        centers=[
            CellCenter(row=0, col=0, x=40.0, y=40.0),
            CellCenter(row=0, col=1, x=120.0, y=40.0),
            CellCenter(row=0, col=2, x=200.0, y=40.0),
            CellCenter(row=0, col=3, x=280.0, y=40.0),
        ],
    )
    observations: dict[str, CellObservation] = {}
    for col in range(4):
        cell_id = _cell_id(0, col)
        crop_path = session_dir / f"{cell_id}.png"
        cv2.imwrite(str(crop_path), np.full((8, 8, 3), 80 + col, dtype=np.uint8))
        observations[cell_id] = CellObservation(
            row=0,
            col=col,
            frame_index=0,
            reveal_score=1.0,
            sharpness=1.0,
            crop_path=str(crop_path),
        )

    groups = [
        MatchGroup(
            label="G001",
            members=[_cell_id(0, 0), _cell_id(0, 1)],
            click_order=[_cell_id(0, 0), _cell_id(0, 1)],
            click_order_positions=[{"row": 0, "col": 0}, {"row": 0, "col": 1}],
            confidence=0.95,
            group_size=2,
        ),
        MatchGroup(
            label="G002",
            members=[_cell_id(0, 2), _cell_id(0, 3)],
            click_order=[_cell_id(0, 2), _cell_id(0, 3)],
            click_order_positions=[{"row": 0, "col": 2}, {"row": 0, "col": 3}],
            confidence=0.93,
            group_size=2,
        ),
    ]
    result = ReconstructionResult(
        calibration=calibration,
        observations=observations,
        groups=groups,
        unresolved=[],
        session_dir=str(session_dir),
    )
    result_path = session_dir / "result.json"
    result.save(result_path)
    return result_path


class AutoPlayTimingTests(unittest.TestCase):
    def test_jittered_delay_clamps_non_negative(self) -> None:
        for _ in range(100):
            self.assertGreaterEqual(_jittered_delay(0.0, 0.010), 0.0)
            value = _jittered_delay(0.035, 0.010)
            self.assertGreaterEqual(value, 0.025)
            self.assertLessEqual(value, 0.045)

    def test_execute_click_moves_then_presses_and_releases(self) -> None:
        controller = DummyController()
        stop_token = DummyStopToken()
        _execute_click(
            controller,
            (111, 222),
            move_settle_s=0.035,
            mouse_down_hold_s=0.045,
            timing_jitter_s=0.0,
            stop_token=stop_token,
        )
        self.assertEqual(
            controller.events,
            [
                ("move", (111, 222)),
                ("press", Button.left),
                ("release", Button.left),
            ],
        )
        self.assertEqual(stop_token.sleeps, [0.035, 0.045])


class AutoPlayGroupFlowTests(unittest.TestCase):
    def test_capture_cell_uses_client_local_origin_for_warp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = _build_result(Path(tmpdir))
            result = ReconstructionResult.load(result_path)
            frame = np.zeros((120, 320, 4), dtype=np.uint8)
            warped = np.full((80, 320, 3), 127, dtype=np.uint8)

            with (
                patch("matchtile.autoplay.mss.mss", return_value=DummyMSSContext(frame)),
                patch("matchtile.autoplay.warp_image_to_board", return_value=warped) as warp_mock,
            ):
                from matchtile.autoplay import _capture_cell

                crop = _capture_cell(
                    result,
                    _cell_id(0, 0),
                    capture_rect=(100, 200, 320, 120),
                    screen_offset=(100, 200),
                )

            self.assertIsNotNone(crop)
            self.assertEqual(crop.shape[:2], (80, 80))
            self.assertEqual(warp_mock.call_args.kwargs["image_origin"], (0, 0))

    def test_no_change_on_first_tile_skips_group_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = _build_result(Path(tmpdir))
            controller = DummyController()
            stop_token = DummyStopToken()
            dummy_crop = np.zeros((8, 8, 3), dtype=np.uint8)

            with (
                patch("matchtile.autoplay.Controller", return_value=controller),
                patch("matchtile.autoplay._capture_cell", side_effect=[dummy_crop] * 6) as capture_mock,
                patch("matchtile.autoplay._tile_unchanged_after_click", side_effect=[True, False, False]),
                patch("matchtile.autoplay._verify_group_change", return_value=True),
            ):
                clicked = auto_click_groups(
                    result_path,
                    min_confidence=0.0,
                    click_delay_s=0.0,
                    move_settle_s=0.0,
                    mouse_down_hold_s=0.0,
                    timing_jitter_s=0.0,
                    verify=True,
                    capture_rect=(0, 0, 320, 120),
                    stop_token=stop_token,
                )

            self.assertEqual(clicked, 3)
            self.assertEqual(capture_mock.call_count, 6)
            self.assertEqual(
                controller.events,
                [
                    ("move", (40, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (200, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (280, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                ],
            )
            self.assertEqual(stop_token.checkpoints, 2)

    def test_no_change_on_later_tile_skips_rest_of_group_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = _build_result(Path(tmpdir))
            controller = DummyController()
            stop_token = DummyStopToken()
            dummy_crop = np.zeros((8, 8, 3), dtype=np.uint8)

            with (
                patch("matchtile.autoplay.Controller", return_value=controller),
                patch("matchtile.autoplay._capture_cell", side_effect=[dummy_crop] * 8) as capture_mock,
                patch("matchtile.autoplay._tile_unchanged_after_click", side_effect=[False, True, False, False]),
                patch("matchtile.autoplay._verify_group_change", return_value=True),
            ):
                clicked = auto_click_groups(
                    result_path,
                    min_confidence=0.0,
                    click_delay_s=0.0,
                    move_settle_s=0.0,
                    mouse_down_hold_s=0.0,
                    timing_jitter_s=0.0,
                    verify=True,
                    capture_rect=(0, 0, 320, 120),
                    stop_token=stop_token,
                )

            self.assertEqual(clicked, 4)
            self.assertEqual(capture_mock.call_count, 8)
            self.assertEqual(
                controller.events,
                [
                    ("move", (40, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (120, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (200, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (280, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                ],
            )
            self.assertEqual(stop_token.checkpoints, 2)

    def test_post_group_verify_failure_does_not_abort_later_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = _build_result(Path(tmpdir))
            controller = DummyController()
            stop_token = DummyStopToken()
            dummy_crop = np.zeros((8, 8, 3), dtype=np.uint8)

            with (
                patch("matchtile.autoplay.Controller", return_value=controller),
                patch("matchtile.autoplay._capture_cell", side_effect=[dummy_crop] * 8),
                patch("matchtile.autoplay._tile_unchanged_after_click", side_effect=[False, False, False, False]),
                patch("matchtile.autoplay._verify_group_change", side_effect=[False, True]),
            ):
                clicked = auto_click_groups(
                    result_path,
                    min_confidence=0.0,
                    click_delay_s=0.0,
                    move_settle_s=0.0,
                    mouse_down_hold_s=0.0,
                    timing_jitter_s=0.0,
                    verify=True,
                    capture_rect=(0, 0, 320, 120),
                    stop_token=stop_token,
                )

            self.assertEqual(clicked, 4)
            self.assertEqual(
                controller.events,
                [
                    ("move", (40, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (120, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (200, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                    ("move", (280, 40)),
                    ("press", Button.left),
                    ("release", Button.left),
                ],
            )


if __name__ == "__main__":
    unittest.main()
