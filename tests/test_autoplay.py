from __future__ import annotations

import unittest
from unittest.mock import call

from pynput.mouse import Button

from matchtile.autoplay import _execute_click, _jittered_delay


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

    def sleep(self, duration_s: float, step_s: float = 0.02) -> None:
        del step_s
        self.sleeps.append(duration_s)


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


if __name__ == "__main__":
    unittest.main()
