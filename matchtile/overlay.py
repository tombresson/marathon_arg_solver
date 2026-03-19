from __future__ import annotations

from dataclasses import dataclass
import signal
import sys
from pathlib import Path

import cv2
import mss
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from pynput import mouse

from matchtile.config import MatchTileConfig
from matchtile.models import MatchGroup, ReconstructionResult
from matchtile.runtime_control import StopToken
from matchtile.vision import cell_center, cell_polygon, warp_image_to_board


@dataclass(slots=True)
class ClickSelection:
    cells: list[str]


class OverlayWidget(QtWidgets.QWidget):
    def __init__(
        self,
        result: ReconstructionResult,
        config: MatchTileConfig,
        live_capture_rect: tuple[int, int, int, int] | None = None,
        screen_offset: tuple[int, int] = (0, 0),
        stop_token: StopToken | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.config = config
        self.live_capture_rect = live_capture_rect
        self.screen_offset = screen_offset
        self.stop_token = stop_token
        self.selection = ClickSelection(cells=[])
        rect = result.calibration.board_rect
        self.setGeometry(rect.x + screen_offset[0], rect.y + screen_offset[1], rect.width, rect.height)
        flags = QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._listener = mouse.Listener(on_click=self._handle_click)
        self._listener.daemon = True
        self._listener.start()
        if self.stop_token:
            self._kill_timer = QtCore.QTimer(self)
            self._kill_timer.timeout.connect(self._check_stop_requested)
            self._kill_timer.start(50)
        self._center_lookup = {(center.row, center.col): (center.x, center.y) for center in result.calibration.centers}

    def _matching_groups(self, selected: list[str]) -> list[MatchGroup]:
        selected_set = set(selected)
        return [group for group in self.result.groups if selected_set.issubset(set(group.members))]

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._listener.running:
            self._listener.stop()
        super().closeEvent(event)

    def _check_stop_requested(self) -> None:
        try:
            if self.stop_token and self.stop_token.stop_requested():
                self.close()
        except KeyboardInterrupt:
            self.close()

    def _cell_center(self, cell_id: str) -> tuple[float, float]:
        obs = self.result.observations[cell_id]
        return self._center_lookup[(obs.row, obs.col)]

    def _nearest_cell(self, global_x: int, global_y: int) -> str | None:
        rect = self.result.calibration.board_rect
        local_x = global_x - (rect.x + self.screen_offset[0])
        local_y = global_y - (rect.y + self.screen_offset[1])
        if local_x < 0 or local_y < 0 or local_x > rect.width or local_y > rect.height:
            return None
        best_cell = None
        best_dist = float("inf")
        for cell_id in self.result.observations:
            cx, cy = self._cell_center(cell_id)
            dist = ((cx - rect.x) - local_x) ** 2 + ((cy - rect.y) - local_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_cell = cell_id
        if best_dist > max(self.result.calibration.pitch_x, self.result.calibration.pitch_y) ** 2:
            return None
        return best_cell

    def _handle_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        if not pressed or button != mouse.Button.left:
            return
        cell_id = self._nearest_cell(x, y)
        if not cell_id or cell_id in self.selection.cells:
            return
        self.selection.cells.append(cell_id)
        matching_groups = self._matching_groups(self.selection.cells)
        exact_group = next((group for group in matching_groups if len(group.members) == len(self.selection.cells)), None)
        if exact_group:
            self._verify_clicks(exact_group)
            self.selection.cells.clear()
        elif not matching_groups or len(self.selection.cells) >= self.config.max_group_size:
            self.selection.cells.clear()
        self.update()

    def _verify_clicks(self, group: MatchGroup) -> None:
        if not self.live_capture_rect:
            return
        left, top, width, height = self.live_capture_rect
        with mss.mss() as sct:
            grabbed = sct.grab({"left": left, "top": top, "width": width, "height": height})
            frame = np.array(grabbed, dtype=np.uint8)[..., :3]
        board = warp_image_to_board(frame, self.result.calibration, image_origin=self.screen_offset)
        changed: list[str] = []
        for cell_id in group.members:
            obs = self.result.observations.get(cell_id)
            if not obs or not obs.crop_path:
                continue
            reference = cv2.imread(obs.crop_path, cv2.IMREAD_COLOR)
            if reference is None:
                continue
            x = int(obs.col * self.result.calibration.pitch_x)
            y = int(obs.row * self.result.calibration.pitch_y)
            w = int(self.result.calibration.pitch_x)
            h = int(self.result.calibration.pitch_y)
            current = board[max(y, 0) : min(y + h, board.shape[0]), max(x, 0) : min(x + w, board.shape[1])]
            if current.size == 0:
                continue
            current = cv2.resize(current, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
            delta = np.mean(np.abs(current.astype(np.float32) - reference.astype(np.float32)))
            if delta > 18.0:
                changed.append(cell_id)
        if len(changed) == len(group.members) and all(cell_id in changed for cell_id in group.members):
            for solved_group in list(self.result.groups):
                if set(solved_group.members) == set(group.members):
                    self.result.groups.remove(solved_group)
                    for cell_id in changed:
                        self.result.observations.pop(cell_id, None)
                    self.result.unresolved = [cell_id for cell_id in self.result.unresolved if cell_id not in changed]
                    break

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        font = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)

        palette = [QtGui.QColor(color) for color in self.config.overlay_palette]
        for index, group in enumerate(self.result.groups):
            color = palette[index % len(palette)]
            alpha = 210 if not group.ambiguous else 120
            fill = QtGui.QColor(color.red(), color.green(), color.blue(), alpha)
            for cell_id in group.members:
                obs = self.result.observations.get(cell_id)
                if not obs:
                    continue
                polygon = cell_polygon(
                    self.result.calibration,
                    obs.row,
                    obs.col,
                    image_origin=(self.result.calibration.board_rect.x, self.result.calibration.board_rect.y),
                )
                qpoints = [QtCore.QPointF(float(point[0]), float(point[1])) for point in polygon]
                qpolygon = QtGui.QPolygonF(qpoints)
                painter.setBrush(fill)
                painter.drawPolygon(qpolygon)
                painter.setPen(QtGui.QColor("black"))
                center_x, center_y = cell_center(self.result.calibration, obs.row, obs.col)
                center_point = QtCore.QPointF(center_x - self.result.calibration.board_rect.x, center_y - self.result.calibration.board_rect.y)
                text_rect = QtCore.QRectF(center_point.x() - 14, center_point.y() - 10, 28, 20)
                painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, f"{group.label[1:]}/{group.group_size}")
                painter.setPen(QtCore.Qt.PenStyle.NoPen)

        if self.selection.cells:
            outline = QtGui.QPen(QtGui.QColor("#FFFFFF"))
            outline.setWidth(2)
            painter.setPen(outline)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for cell_id in self.selection.cells:
                obs = self.result.observations.get(cell_id)
                if not obs:
                    continue
                polygon = cell_polygon(
                    self.result.calibration,
                    obs.row,
                    obs.col,
                    image_origin=(self.result.calibration.board_rect.x, self.result.calibration.board_rect.y),
                )
                qpolygon = QtGui.QPolygonF([QtCore.QPointF(float(point[0]), float(point[1])) for point in polygon])
                painter.drawPolygon(qpolygon)

        painter.setPen(QtGui.QColor("#FFFFFF"))
        painter.setBrush(QtGui.QColor(10, 10, 10, 180))
        hud = QtCore.QRectF(8, 8, 230, 72)
        painter.drawRoundedRect(hud, 10, 10)
        painter.drawText(
            hud.adjusted(12, 10, -12, -12),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            f"Groups: {len(self.result.groups)}\nUnresolved: {len(self.result.unresolved)}\nSelected: {len(self.selection.cells)}",
        )


def launch_overlay(
    result_path: Path,
    config: MatchTileConfig,
    live_capture_rect: tuple[int, int, int, int] | None = None,
    screen_offset: tuple[int, int] = (0, 0),
    stop_token: StopToken | None = None,
) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    result = ReconstructionResult.load(result_path)
    widget = OverlayWidget(
        result=result,
        config=config,
        live_capture_rect=live_capture_rect,
        screen_offset=screen_offset,
        stop_token=stop_token,
    )
    widget.show()

    previous_sigint = None
    previous_sigbreak = None

    def _quit_from_signal(signum: int, _frame: object) -> None:
        del signum
        widget.close()
        app.quit()

    try:
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _quit_from_signal)
        if hasattr(signal, "SIGBREAK"):
            previous_sigbreak = signal.getsignal(signal.SIGBREAK)
            signal.signal(signal.SIGBREAK, _quit_from_signal)
        return app.exec()
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        if previous_sigbreak is not None and hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, previous_sigbreak)
