from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import win32con
import win32gui

from matchtile.models import Rect


@dataclass(slots=True)
class WindowTarget:
    hwnd: int
    title: str
    client_rect: Rect


def _enum_visible_windows() -> Iterable[tuple[int, str]]:
    items: list[tuple[int, str]] = []

    def collect(hwnd: int, _: object) -> None:
        title = win32gui.GetWindowText(hwnd)
        if not title or not win32gui.IsWindowVisible(hwnd):
            return
        if win32gui.IsIconic(hwnd):
            return
        items.append((hwnd, title))

    win32gui.EnumWindows(collect, None)
    return items


def find_window(title_regex: str) -> WindowTarget:
    pattern = re.compile(title_regex, re.IGNORECASE)
    for hwnd, title in _enum_visible_windows():
        if pattern.search(title):
            client_rect = get_client_rect(hwnd)
            return WindowTarget(hwnd=hwnd, title=title, client_rect=client_rect)
    raise RuntimeError(f"No visible window matched regex: {title_regex}")


def get_client_rect(hwnd: int) -> Rect:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (left, top))
    return Rect(x=x, y=y, width=right - left, height=bottom - top)


def bring_window_to_front(hwnd: int) -> None:
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
