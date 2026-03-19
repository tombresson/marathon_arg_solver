from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput import keyboard

from matchtile.models import Rect
from matchtile.runtime_control import StopToken


@dataclass(slots=True)
class CaptureMetadata:
    fps: int
    duration_s: float
    frame_count: int
    capture_rect: Rect


def wait_for_hotkey(start_key: keyboard.Key = keyboard.Key.f8, stop_token: StopToken | None = None) -> None:
    print(f"Press {start_key.name.upper()} to start. Press F12 to abort.")
    started = False

    def on_press(key: keyboard.KeyCode | keyboard.Key) -> bool | None:
        nonlocal started
        if key == start_key:
            started = True
            return False
        return None

    with keyboard.Listener(on_press=on_press) as listener:
        while listener.is_alive():
            listener.join(0.05)
            if started:
                break
            if stop_token:
                stop_token.checkpoint()
    print(f"{start_key.name.upper()} received. Starting now.")


def capture_frame(rect: Rect) -> np.ndarray:
    with mss.mss() as sct:
        grabbed = sct.grab({"left": rect.x, "top": rect.y, "width": rect.width, "height": rect.height})
        return np.array(grabbed, dtype=np.uint8)[..., :3]


def capture_frames(rect: Rect, fps: int, duration_s: float, out_dir: Path, stop_token: StopToken | None = None) -> CaptureMetadata:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_interval = 1.0 / max(fps, 1)
    count = 0
    with mss.mss() as sct:
        started = time.perf_counter()
        deadline = started + duration_s
        while time.perf_counter() < deadline:
            if stop_token:
                stop_token.checkpoint()
            grabbed = sct.grab({"left": rect.x, "top": rect.y, "width": rect.width, "height": rect.height})
            image = np.array(grabbed, dtype=np.uint8)[..., :3]
            cv2.imwrite(str(frames_dir / f"frame_{count:04d}.png"), image)
            count += 1
            target = started + count * frame_interval
            remaining = target - time.perf_counter()
            if remaining > 0:
                if stop_token:
                    stop_token.sleep(remaining)
                else:
                    time.sleep(remaining)
    return CaptureMetadata(fps=fps, duration_s=duration_s, frame_count=count, capture_rect=rect)
