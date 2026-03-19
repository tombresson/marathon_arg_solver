from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

from pynput import keyboard


class AbortRequested(RuntimeError):
    """Raised when the global stop hotkey is pressed."""


@dataclass(slots=True)
class StopToken:
    stop_key: keyboard.Key = keyboard.Key.f12
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _listener: keyboard.Listener | None = None

    def __enter__(self) -> "StopToken":
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._listener and self._listener.running:
            self._listener.stop()
        self._listener = None

    def _on_press(self, key: keyboard.KeyCode | keyboard.Key) -> bool | None:
        if key == self.stop_key:
            self._stop_event.set()
            return False
        return None

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def checkpoint(self) -> None:
        if self.stop_requested():
            raise AbortRequested(f"Emergency stop requested via {self.stop_key.name.upper()}.")

    def sleep(self, duration_s: float, step_s: float = 0.02) -> None:
        deadline = time.perf_counter() + max(duration_s, 0.0)
        while True:
            self.checkpoint()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(step_s, remaining))
