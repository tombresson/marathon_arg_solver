from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import win32con
import win32gui

from matchtile.config import MatchTileConfig
from matchtile.debug_grid import render_candidate_debug, render_debug_grids
from matchtile.models import Calibration, CellCenter, CellObservation, MatchGroup, Point, Rect, ReconstructionResult
from matchtile.windowing import bring_window_to_front


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def detect_discord_activity_region(image: np.ndarray) -> Rect:
    height, width = image.shape[:2]
    return Rect(0, 0, width, height)


def crop_rect(image: np.ndarray, rect: Rect) -> np.ndarray:
    return image[rect.y : rect.bottom, rect.x : rect.right].copy()


def manual_select_rect(image: np.ndarray, title: str = "Select board ROI") -> Rect:
    x, y, width, height = cv2.selectROI(title, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(title)
    if width <= 0 or height <= 0:
        raise RuntimeError("Manual selection was cancelled.")
    return Rect(int(x), int(y), int(width), int(height))


def _point_array(points: list[Point]) -> np.ndarray:
    return np.array([[point.x, point.y] for point in points], dtype=np.float32)


def _board_rect_from_corners(corners: list[Point]) -> Rect:
    xs = [point.x for point in corners]
    ys = [point.y for point in corners]
    left = int(np.floor(min(xs)))
    top = int(np.floor(min(ys)))
    right = int(np.ceil(max(xs)))
    bottom = int(np.ceil(max(ys)))
    return Rect(left, top, max(right - left, 1), max(bottom - top, 1))


def manual_select_corners(
    image: np.ndarray,
    rows: int,
    cols: int,
    title: str = "Click board corners",
) -> tuple[list[Point], int, int]:
    window_name = title
    points: list[Point] = []
    current_rows = max(1, int(rows))
    current_cols = max(1, int(cols))
    instructions = [
        "Click inside this calibration image window.",
        "Click top-left, top-right, bottom-right, bottom-left.",
        "After 4 corners: arrows adjust counts, Enter saves, Backspace undoes, R resets, Esc cancels.",
    ]

    def focus_window() -> None:
        try:
            hwnd = win32gui.FindWindow(None, window_name)
            if not hwnd:
                return
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
            bring_window_to_front(hwnd)
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
        except Exception:
            return

    def redraw() -> None:
        canvas = image.copy()
        calibration: Calibration | None = None
        if len(points) == 4:
            calibration = build_manual_calibration(
                (image.shape[1], image.shape[0]),
                rows=current_rows,
                cols=current_cols,
                corners=points,
            )
            canvas = _render_grid_fit_overlay(canvas, calibration)
        for index, line in enumerate(instructions):
            cv2.putText(canvas, line, (16, 30 + index * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        progress = f"Captured corners: {len(points)}/4"
        cv2.putText(canvas, progress, (16, 30 + len(instructions) * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        count_line = f"Counts: {current_cols} cols x {current_rows} rows"
        cv2.putText(
            canvas,
            count_line,
            (16, 30 + (len(instructions) + 1) * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labels = ["TL", "TR", "BR", "BL"]
        for index, point in enumerate(points):
            xy = (int(round(point.x)), int(round(point.y)))
            cv2.circle(canvas, xy, 6, (0, 255, 0), -1)
            cv2.putText(canvas, labels[index], (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        if len(points) >= 2:
            poly = np.array([[int(round(point.x)), int(round(point.y))] for point in points], dtype=np.int32)
            cv2.polylines(canvas, [poly], False, (0, 255, 0), 2, cv2.LINE_AA)
        if len(points) == 4:
            poly = np.array([[int(round(point.x)), int(round(point.y))] for point in points], dtype=np.int32)
            cv2.polylines(canvas, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append(Point(float(x), float(y)))
            labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
            print(f"Captured {labels[len(points) - 1]} corner at ({x}, {y}).")
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    cv2.waitKey(1)
    focus_window()
    print("Calibration window is active. Click the four board corners inside that window.")
    try:
        while True:
            key = cv2.waitKeyEx(20)
            if key in (13, 10, 32) and len(points) == 4:
                return points.copy(), current_rows, current_cols
            if key in (8, 127) and points:
                removed = points.pop()
                print(f"Removed last corner at ({int(round(removed.x))}, {int(round(removed.y))}).")
                redraw()
            if key in (ord("r"), ord("R")):
                if points:
                    print("Resetting captured corners.")
                points.clear()
                redraw()
            if len(points) == 4 and key == 2424832:
                next_cols = max(1, current_cols - 1)
                if next_cols != current_cols:
                    current_cols = next_cols
                    print(f"Columns set to {current_cols}.")
                    redraw()
            if len(points) == 4 and key == 2555904:
                current_cols += 1
                print(f"Columns set to {current_cols}.")
                redraw()
            if len(points) == 4 and key == 2490368:
                current_rows += 1
                print(f"Rows set to {current_rows}.")
                redraw()
            if len(points) == 4 and key == 2621440:
                next_rows = max(1, current_rows - 1)
                if next_rows != current_rows:
                    current_rows = next_rows
                    print(f"Rows set to {current_rows}.")
                    redraw()
            if key == 27:
                raise RuntimeError("Manual corner selection was cancelled.")
    finally:
        cv2.destroyWindow(window_name)


def _edge_length(left: Point, right: Point) -> float:
    return float(np.hypot(right.x - left.x, right.y - left.y))


def build_manual_calibration(
    image_size: tuple[int, int],
    rows: int,
    cols: int,
    corners: list[Point],
    source: str = "manual-corners",
) -> Calibration:
    if len(corners) != 4:
        raise ValueError("Manual calibration requires exactly 4 corners.")
    image_width, image_height = image_size
    board_rect = _board_rect_from_corners(corners)
    top_len = _edge_length(corners[0], corners[1])
    bottom_len = _edge_length(corners[3], corners[2])
    left_len = _edge_length(corners[0], corners[3])
    right_len = _edge_length(corners[1], corners[2])
    pitch_x = max(12.0, (top_len + bottom_len) / max(2 * cols, 1))
    pitch_y = max(12.0, (left_len + right_len) / max(2 * rows, 1))
    rectified_width = max(int(round(pitch_x * cols)), cols)
    rectified_height = max(int(round(pitch_y * rows)), rows)
    pitch_x = rectified_width / max(cols, 1)
    pitch_y = rectified_height / max(rows, 1)

    destination_corners = np.array(
        [[0.0, 0.0], [rectified_width, 0.0], [rectified_width, rectified_height], [0.0, rectified_height]],
        dtype=np.float32,
    )
    board_to_source = cv2.getPerspectiveTransform(destination_corners, _point_array(corners))
    board_centers = np.array(
        [[[(col + 0.5) * pitch_x, (row + 0.5) * pitch_y]] for row in range(rows) for col in range(cols)],
        dtype=np.float32,
    )
    projected_centers = cv2.perspectiveTransform(board_centers, board_to_source).reshape(-1, 2)
    centers = [
        CellCenter(row=row, col=col, x=float(projected_centers[row * cols + col][0]), y=float(projected_centers[row * cols + col][1]))
        for row in range(rows)
        for col in range(cols)
    ]

    return Calibration(
        board_rect=board_rect,
        rows=rows,
        cols=cols,
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
        offset_x=0.0,
        offset_y=0.0,
        board_corners=[Point(point.x, point.y) for point in corners],
        client_width=int(image_width),
        client_height=int(image_height),
        rectified_width=int(rectified_width),
        rectified_height=int(rectified_height),
        source=source,
        centers=centers,
    )


def calibrate_grid(image: np.ndarray, rows: int, cols: int, corners: list[Point], source: str = "manual-corners") -> Calibration:
    return build_manual_calibration((image.shape[1], image.shape[0]), rows=rows, cols=cols, corners=corners, source=source)


def _calibration_source_corners(calibration: Calibration, image_origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    origin_x, origin_y = image_origin
    return np.array(
        [[point.x - origin_x, point.y - origin_y] for point in calibration.board_corners],
        dtype=np.float32,
    )


def _board_destination_corners(calibration: Calibration) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(calibration.rectified_width), 0.0],
            [float(calibration.rectified_width), float(calibration.rectified_height)],
            [0.0, float(calibration.rectified_height)],
        ],
        dtype=np.float32,
    )


def warp_image_to_board(image: np.ndarray, calibration: Calibration, image_origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    source = _calibration_source_corners(calibration, image_origin=image_origin)
    destination = _board_destination_corners(calibration)
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (calibration.rectified_width, calibration.rectified_height))


def project_board_points(
    calibration: Calibration,
    board_points: np.ndarray,
    image_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    source = _calibration_source_corners(calibration, image_origin=image_origin)
    destination = _board_destination_corners(calibration)
    matrix = cv2.getPerspectiveTransform(destination, source)
    points = np.asarray(board_points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 2)


def cell_polygon(calibration: Calibration, row: int, col: int, image_origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    x0 = col * calibration.pitch_x
    y0 = row * calibration.pitch_y
    polygon = np.array(
        [
            [x0, y0],
            [x0 + calibration.pitch_x, y0],
            [x0 + calibration.pitch_x, y0 + calibration.pitch_y],
            [x0, y0 + calibration.pitch_y],
        ],
        dtype=np.float32,
    )
    return project_board_points(calibration, polygon, image_origin=image_origin)


def cell_center(calibration: Calibration, row: int, col: int) -> tuple[float, float]:
    index = row * calibration.cols + col
    center = calibration.centers[index]
    return center.x, center.y


def offset_calibration(calibration: Calibration, dx: float, dy: float) -> Calibration:
    shifted_corners = [Point(point.x + dx, point.y + dy) for point in calibration.board_corners]
    shifted_centers = [CellCenter(row=center.row, col=center.col, x=center.x + dx, y=center.y + dy) for center in calibration.centers]
    return Calibration(
        board_rect=Rect(calibration.board_rect.x + int(round(dx)), calibration.board_rect.y + int(round(dy)), calibration.board_rect.width, calibration.board_rect.height),
        rows=calibration.rows,
        cols=calibration.cols,
        pitch_x=calibration.pitch_x,
        pitch_y=calibration.pitch_y,
        offset_x=calibration.offset_x,
        offset_y=calibration.offset_y,
        board_corners=shifted_corners,
        client_width=calibration.client_width,
        client_height=calibration.client_height,
        rectified_width=calibration.rectified_width,
        rectified_height=calibration.rectified_height,
        source=calibration.source,
        centers=shifted_centers,
    )


def _cell_crop(image: np.ndarray, calibration: Calibration, row: int, col: int, inset: float = 0.18) -> np.ndarray | None:
    center_x = calibration.offset_x + (col + 0.5) * calibration.pitch_x
    center_y = calibration.offset_y + (row + 0.5) * calibration.pitch_y
    tile_w = calibration.pitch_x * (1.0 - inset)
    tile_h = calibration.pitch_y * (1.0 - inset)
    x0 = int(max(center_x - tile_w / 2, 0))
    y0 = int(max(center_y - tile_h / 2, 0))
    x1 = int(min(center_x + tile_w / 2, image.shape[1]))
    y1 = int(min(center_y + tile_h / 2, image.shape[0]))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image[y0:y1, x0:x1].copy()


def _reveal_score(crop: np.ndarray) -> tuple[float, float]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat_mean = float(hsv[..., 1].mean())
    val_mean = float(hsv[..., 2].mean())
    std_gray = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    score = sat_mean * 0.8 + val_mean * 0.2 + std_gray * 1.2
    return score, sharpness


@dataclass(slots=True)
class CellFrameSample:
    frame_index: int
    crop: np.ndarray
    reveal_score: float
    sharpness: float
    state: str = "hidden"
    visible_width_ratio: float = 0.0
    occupancy_ratio: float = 0.0
    quality_score: float = 0.0
    plateau_len: int = 0


def _build_hidden_reference(samples: list[CellFrameSample]) -> np.ndarray:
    count = min(3, len(samples))
    lowest = sorted(samples, key=lambda sample: sample.reveal_score)[:count]
    stack = np.stack([sample.crop.astype(np.float32) for sample in lowest], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _classify_sample_state(sample: CellFrameSample, hidden_reference: np.ndarray) -> None:
    crop = sample.crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hidden_gray = cv2.cvtColor(hidden_reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    diff = cv2.absdiff(crop, hidden_reference)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sat = hsv[..., 1] / 255.0
    edges = cv2.Canny(gray.astype(np.uint8), 40, 140).astype(np.float32) / 255.0
    intensity_delta = np.abs(gray - hidden_gray) / 255.0
    activity = np.maximum.reduce([diff_gray * 1.25, intensity_delta, sat * 0.85, edges * 0.70])
    mask = activity > 0.16

    occupancy_ratio = float(mask.mean())
    column_activity = mask.mean(axis=0)
    active_threshold = max(0.10, float(column_activity.max()) * 0.35)
    active_cols = np.where(column_activity >= active_threshold)[0]
    if active_cols.size:
        first_col = int(active_cols[0])
        last_col = int(active_cols[-1])
        visible_width_ratio = float((last_col - first_col + 1) / max(crop.shape[1], 1))
    else:
        first_col = 0
        last_col = -1
        visible_width_ratio = 0.0

    quarter = max(1, crop.shape[1] // 4)
    center_band = mask[:, quarter : crop.shape[1] - quarter]
    edge_band = np.concatenate([mask[:, :quarter], mask[:, crop.shape[1] - quarter :]], axis=1)
    center_occupancy = float(center_band.mean()) if center_band.size else 0.0
    edge_occupancy = float(edge_band.mean()) if edge_band.size else 0.0
    left_margin_ratio = float(first_col / max(crop.shape[1], 1)) if active_cols.size else 1.0
    right_margin_ratio = float((crop.shape[1] - 1 - last_col) / max(crop.shape[1], 1)) if active_cols.size else 1.0

    sample.visible_width_ratio = round(visible_width_ratio, 4)
    sample.occupancy_ratio = round(occupancy_ratio, 4)

    if occupancy_ratio < 0.08 and sample.reveal_score < 70.0:
        sample.state = "hidden"
        return

    if (
        visible_width_ratio >= 0.72
        and occupancy_ratio >= 0.11
        and edge_occupancy >= center_occupancy * 0.16
    ):
        sample.state = "fully-revealed"
        return

    if (
        visible_width_ratio >= 0.12
        and (left_margin_ratio >= 0.08 or right_margin_ratio >= 0.08)
        and center_occupancy > edge_occupancy * 1.20
    ):
        sample.state = "transition"
        return

    sample.state = "hidden" if sample.reveal_score < 85.0 else "transition"


def _refine_transition_states(samples: list[CellFrameSample]) -> None:
    widths = [sample.visible_width_ratio for sample in samples]
    for index, sample in enumerate(samples):
        if sample.state != "transition":
            continue
        prev_width = widths[index - 1] if index > 0 else sample.visible_width_ratio
        next_width = widths[index + 1] if index + 1 < len(samples) else sample.visible_width_ratio
        if next_width >= prev_width:
            sample.state = "transition-opening"
        else:
            sample.state = "transition-closing"


def _apply_plateau_scores(samples: list[CellFrameSample]) -> None:
    for index, sample in enumerate(samples):
        if sample.state != "fully-revealed":
            continue
        left = index
        right = index
        while left - 1 >= 0 and samples[left - 1].state == "fully-revealed":
            left -= 1
        while right + 1 < len(samples) and samples[right + 1].state == "fully-revealed":
            right += 1
        sample.plateau_len = right - left + 1
        plateau_score = min(sample.plateau_len / 3.0, 1.0)
        reveal_norm = min(sample.reveal_score / 220.0, 1.0)
        sample.quality_score = float(
            0.38 * sample.visible_width_ratio
            + 0.28 * sample.occupancy_ratio
            + 0.24 * plateau_score
            + 0.10 * reveal_norm
        )


def _candidate_record(sample: CellFrameSample, crop_path: str | None) -> dict:
    return {
        "frame_index": sample.frame_index,
        "state": sample.state,
        "reveal_score": round(sample.reveal_score, 3),
        "sharpness": round(sample.sharpness, 3),
        "visible_width_ratio": round(sample.visible_width_ratio, 4),
        "occupancy_ratio": round(sample.occupancy_ratio, 4),
        "quality_score": round(sample.quality_score, 4),
        "crop_path": crop_path,
    }


def _timeline_record(sample: CellFrameSample) -> dict:
    return {
        "frame_index": sample.frame_index,
        "state": sample.state,
        "reveal_score": round(sample.reveal_score, 3),
        "sharpness": round(sample.sharpness, 3),
        "visible_width_ratio": round(sample.visible_width_ratio, 4),
        "occupancy_ratio": round(sample.occupancy_ratio, 4),
        "quality_score": round(sample.quality_score, 4),
    }


def _write_crop(path: Path, crop: np.ndarray) -> str:
    cv2.imwrite(str(path), crop)
    return str(path)


@dataclass(slots=True)
class TileFeature:
    cell_id: str
    descriptor: np.ndarray
    shape_descriptor: np.ndarray | None = None
    color_signature: np.ndarray | None = None
    alternate_descriptors: list[np.ndarray] = field(default_factory=list)
    alternate_shape_descriptors: list[np.ndarray] = field(default_factory=list)
    alternate_color_signatures: list[np.ndarray] = field(default_factory=list)
    variant_descriptors: list[np.ndarray] = field(default_factory=list)


def _combine_feature_variant(
    descriptor: np.ndarray,
    shape_descriptor: np.ndarray | None,
    color_signature: np.ndarray | None,
) -> np.ndarray:
    parts = [descriptor.astype(np.float32, copy=False) * 0.28]
    if shape_descriptor is not None:
        parts.append(shape_descriptor.astype(np.float32, copy=False) * 0.42)
    if color_signature is not None:
        parts.append(color_signature.astype(np.float32, copy=False) * 1.10)
    if len(parts) == 1:
        combined = descriptor.astype(np.float32, copy=False)
    else:
        combined = np.concatenate(parts, axis=0)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined.astype(np.float32, copy=False)


def _extract_symbol_mask(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    border = np.concatenate(
        [
            resized[:6, :, :].reshape(-1, 3),
            resized[-6:, :, :].reshape(-1, 3),
            resized[:, :6, :].reshape(-1, 3),
            resized[:, -6:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border.astype(np.float32), axis=0)
    color_delta = np.linalg.norm(resized.astype(np.float32) - background[None, None, :], axis=2) / 255.0
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background_gray = float(np.median(cv2.cvtColor(border.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY)))
    intensity_delta = np.abs(gray - background_gray) / 255.0
    edges = cv2.Canny(gray.astype(np.uint8), 40, 140).astype(np.float32) / 255.0
    activity = np.maximum.reduce([color_delta * 0.85, intensity_delta * 1.20, edges * 0.80])
    threshold = max(0.14, float(np.quantile(activity, 0.82)) * 0.72)
    mask = (activity >= threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _build_shape_descriptor(crop: np.ndarray) -> np.ndarray:
    mask = _extract_symbol_mask(crop)
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((24 * 24,), dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    symbol = mask[y0:y1, x0:x1]
    side = max(symbol.shape[0], symbol.shape[1], 1)
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - symbol.shape[0]) // 2
    offset_x = (side - symbol.shape[1]) // 2
    canvas[offset_y : offset_y + symbol.shape[0], offset_x : offset_x + symbol.shape[1]] = symbol
    normalized = cv2.resize(canvas, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    descriptor = normalized.reshape(-1)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    return descriptor


def _build_color_signature(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    mask = _extract_symbol_mask(crop) > 0
    if not np.any(mask):
        return np.zeros((6,), dtype=np.float32)
    foreground = resized[mask].astype(np.float32) / 255.0
    background = resized[~mask].astype(np.float32) / 255.0
    if background.size == 0:
        background = resized.reshape(-1, 3).astype(np.float32) / 255.0
    signature = np.concatenate([foreground.mean(axis=0), background.mean(axis=0)], axis=0).astype(np.float32)
    norm = np.linalg.norm(signature)
    if norm > 0:
        signature = signature / norm
    return signature


def _build_feature(cell_id: str, crop: np.ndarray, alternate_crops: list[np.ndarray] | None = None) -> TileFeature:
    resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, None).flatten()
    edge = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 60, 160).astype(np.float32).reshape(-1) / 255.0
    thumb = cv2.resize(resized, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
    descriptor = np.concatenate([thumb, hist.astype(np.float32), edge], axis=0)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    shape_descriptor = _build_shape_descriptor(crop)
    color_signature = _build_color_signature(crop)
    alternate_descriptors: list[np.ndarray] = []
    alternate_shape_descriptors: list[np.ndarray] = []
    alternate_color_signatures: list[np.ndarray] = []
    for alt_crop in alternate_crops or []:
        alt_resized = cv2.resize(alt_crop, (32, 32), interpolation=cv2.INTER_AREA)
        alt_hsv = cv2.cvtColor(alt_resized, cv2.COLOR_BGR2HSV)
        alt_hist = cv2.calcHist([alt_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        alt_hist = cv2.normalize(alt_hist, None).flatten()
        alt_edge = cv2.Canny(cv2.cvtColor(alt_resized, cv2.COLOR_BGR2GRAY), 60, 160).astype(np.float32).reshape(-1) / 255.0
        alt_thumb = cv2.resize(alt_resized, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1) / 255.0
        alt_descriptor = np.concatenate([alt_thumb, alt_hist.astype(np.float32), alt_edge], axis=0)
        alt_norm = np.linalg.norm(alt_descriptor)
        if alt_norm > 0:
            alt_descriptor = alt_descriptor / alt_norm
        alternate_descriptors.append(alt_descriptor)
        alternate_shape_descriptors.append(_build_shape_descriptor(alt_crop))
        alternate_color_signatures.append(_build_color_signature(alt_crop))
    variant_descriptors = [_combine_feature_variant(descriptor, shape_descriptor, color_signature)]
    for index, alt_descriptor in enumerate(alternate_descriptors):
        alt_shape = alternate_shape_descriptors[index] if index < len(alternate_shape_descriptors) else None
        alt_color = alternate_color_signatures[index] if index < len(alternate_color_signatures) else None
        variant_descriptors.append(_combine_feature_variant(alt_descriptor, alt_shape, alt_color))
    return TileFeature(
        cell_id=cell_id,
        descriptor=descriptor,
        shape_descriptor=shape_descriptor,
        color_signature=color_signature,
        alternate_descriptors=alternate_descriptors,
        alternate_shape_descriptors=alternate_shape_descriptors,
        alternate_color_signatures=alternate_color_signatures,
        variant_descriptors=variant_descriptors,
    )


def _feature_variants(feature: TileFeature) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None]]:
    variants = [(feature.descriptor, feature.shape_descriptor, feature.color_signature)]
    for index, descriptor in enumerate(feature.alternate_descriptors):
        shape_descriptor = feature.alternate_shape_descriptors[index] if index < len(feature.alternate_shape_descriptors) else None
        color_signature = feature.alternate_color_signatures[index] if index < len(feature.alternate_color_signatures) else None
        variants.append((descriptor, shape_descriptor, color_signature))
    return variants


def compare_features(left: TileFeature, right: TileFeature) -> float:
    best = 0.0
    for left_descriptor, left_shape, left_color in _feature_variants(left):
        for right_descriptor, right_shape, right_color in _feature_variants(right):
            appearance = float(np.clip(np.dot(left_descriptor, right_descriptor), 0.0, 1.0))
            shape = float(np.clip(np.dot(left_shape, right_shape), 0.0, 1.0)) if left_shape is not None and right_shape is not None else appearance
            color = float(np.clip(np.dot(left_color, right_color), 0.0, 1.0)) if left_color is not None and right_color is not None else appearance
            score = 0.20 * appearance + 0.80 * (shape * color)
            if score > best:
                best = score
    return best


def _render_grid_fit_overlay(image: np.ndarray, calibration: Calibration) -> np.ndarray:
    overlay = image.copy()
    board_polygon = np.array(
        [[int(round(point.x)), int(round(point.y))] for point in calibration.board_corners],
        dtype=np.int32,
    )
    cv2.polylines(overlay, [board_polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
    for col in range(calibration.cols + 1):
        top, bottom = project_board_points(
            calibration,
            np.array([[col * calibration.pitch_x, 0.0], [col * calibration.pitch_x, calibration.rectified_height]], dtype=np.float32),
        )
        cv2.line(
            overlay,
            (int(round(top[0])), int(round(top[1]))),
            (int(round(bottom[0])), int(round(bottom[1]))),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    for row in range(calibration.rows + 1):
        left, right = project_board_points(
            calibration,
            np.array([[0.0, row * calibration.pitch_y], [calibration.rectified_width, row * calibration.pitch_y]], dtype=np.float32),
        )
        cv2.line(
            overlay,
            (int(round(left[0])), int(round(left[1]))),
            (int(round(right[0])), int(round(right[1]))),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    label = (
        f"{calibration.cols} cols x {calibration.rows} rows | "
        f"pitch {calibration.pitch_x:.2f} x {calibration.pitch_y:.2f}"
    )
    label_x = calibration.board_rect.x
    label_y = max(24, calibration.board_rect.y - 12)
    cv2.putText(overlay, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def render_grid_fit_debug(image: np.ndarray, calibration: Calibration, output_path: Path) -> Path:
    overlay = _render_grid_fit_overlay(image, calibration)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def _ordered_group_members(members: list[str], observations: dict[str, CellObservation]) -> tuple[list[str], list[dict]]:
    ordered = sorted(
        members,
        key=lambda cell_id: (
            observations[cell_id].row if cell_id in observations else 9999,
            observations[cell_id].col if cell_id in observations else 9999,
            cell_id,
        ),
    )
    positions = [
        {
            "cell_id": cell_id,
            "row": observations[cell_id].row,
            "col": observations[cell_id].col,
        }
        for cell_id in ordered
        if cell_id in observations
    ]
    return ordered, positions


def reconstruct_from_images(
    image_paths: list[Path],
    session_dir: Path,
    config: MatchTileConfig,
    calibration: Calibration,
) -> ReconstructionResult:
    images = [load_image(path) for path in image_paths]
    if not images:
        raise ValueError("No images provided for reconstruction.")

    calibration_image = images[0]
    boards = [warp_image_to_board(frame, calibration) for frame in images]
    observations: dict[str, CellObservation] = {}
    features: dict[str, TileFeature] = {}
    deferred_unresolved: set[str] = set()

    crops_dir = session_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for row in range(calibration.rows):
        for col in range(calibration.cols):
            cell_id = f"r{row:02d}c{col:02d}"
            samples: list[CellFrameSample] = []
            for frame_index, board in enumerate(boards):
                crop = _cell_crop(board, calibration, row, col)
                if crop is None:
                    continue
                reveal_score, sharpness = _reveal_score(crop)
                samples.append(
                    CellFrameSample(
                        frame_index=frame_index,
                        crop=crop,
                        reveal_score=reveal_score,
                        sharpness=sharpness,
                    )
                )
            if not samples:
                continue

            hidden_reference = _build_hidden_reference(samples)
            for sample in samples:
                _classify_sample_state(sample, hidden_reference)
            _refine_transition_states(samples)
            _apply_plateau_scores(samples)

            stable_candidates = [sample for sample in samples if sample.state == "fully-revealed"]
            transition_candidates = [
                sample for sample in samples if sample.state in {"transition-opening", "transition-closing"}
            ]

            timeline = [_timeline_record(sample) for sample in samples if sample.state != "hidden"]

            if stable_candidates:
                stable_candidates.sort(
                    key=lambda sample: (sample.quality_score, sample.visible_width_ratio, sample.reveal_score),
                    reverse=True,
                )
                primary = stable_candidates[0]
                primary_path = _write_crop(crops_dir / f"{cell_id}.png", primary.crop)
                alternates: list[dict] = []
                for alt_index, sample in enumerate(stable_candidates[1:3], start=1):
                    alt_path = _write_crop(crops_dir / f"{cell_id}__alt{alt_index}_f{sample.frame_index:04d}.png", sample.crop)
                    alternates.append(_candidate_record(sample, alt_path))
                discarded: list[dict] = []
                for drop_index, sample in enumerate(
                    sorted(transition_candidates, key=lambda item: (item.reveal_score, item.visible_width_ratio), reverse=True)[:2],
                    start=1,
                ):
                    drop_path = _write_crop(crops_dir / f"{cell_id}__drop{drop_index}_f{sample.frame_index:04d}.png", sample.crop)
                    discarded.append(_candidate_record(sample, drop_path))

                observations[cell_id] = CellObservation(
                    row=row,
                    col=col,
                    frame_index=primary.frame_index,
                    reveal_score=round(primary.reveal_score, 3),
                    sharpness=round(primary.sharpness, 3),
                    crop_path=primary_path,
                    state=primary.state,
                    quality_score=round(primary.quality_score, 4),
                    visible_width_ratio=round(primary.visible_width_ratio, 4),
                    occupancy_ratio=round(primary.occupancy_ratio, 4),
                    alternate_candidates=alternates,
                    discarded_candidates=discarded,
                    timeline=timeline,
                )
                features[cell_id] = _build_feature(
                    cell_id,
                    primary.crop,
                    alternate_crops=[sample.crop for sample in stable_candidates[1:3]],
                )
            elif transition_candidates:
                discarded = []
                for drop_index, sample in enumerate(
                    sorted(transition_candidates, key=lambda item: (item.reveal_score, item.visible_width_ratio), reverse=True)[:3],
                    start=1,
                ):
                    drop_path = _write_crop(crops_dir / f"{cell_id}__drop{drop_index}_f{sample.frame_index:04d}.png", sample.crop)
                    discarded.append(_candidate_record(sample, drop_path))
                observations[cell_id] = CellObservation(
                    row=row,
                    col=col,
                    frame_index=-1,
                    reveal_score=0.0,
                    sharpness=0.0,
                    crop_path=None,
                    state="unresolved",
                    quality_score=0.0,
                    visible_width_ratio=0.0,
                    occupancy_ratio=0.0,
                    alternate_candidates=[],
                    discarded_candidates=discarded,
                    timeline=timeline,
                )
                deferred_unresolved.add(cell_id)

    groups, unresolved = infer_match_groups(features, config, observations=observations)
    unresolved = sorted(set(unresolved).union(deferred_unresolved))
    result = ReconstructionResult(
        calibration=calibration,
        observations=observations,
        groups=groups,
        unresolved=unresolved,
        session_dir=str(session_dir),
    )
    fit_debug_path = render_grid_fit_debug(calibration_image, calibration, session_dir / "grid_fit_debug.png")
    result.grid_fit_debug_path = str(fit_debug_path)
    composed_path, debug_path = render_debug_grids(result, session_dir)
    result.grid_composed_path = str(composed_path)
    result.grid_debug_path = str(debug_path)
    candidate_debug_path = render_candidate_debug(result, session_dir)
    result.candidate_debug_path = str(candidate_debug_path) if candidate_debug_path else None
    return result


def _candidate_confidence(
    similarity_matrix: np.ndarray,
    group_indices: tuple[int, ...],
    neighbor_rankings: list[list[int]],
    config: MatchTileConfig,
) -> tuple[float, float, float, float, float] | None:
    size = len(group_indices)
    internal_values = [
        float(similarity_matrix[left, right])
        for left, right in combinations(group_indices, 2)
    ]
    if not internal_values:
        return None
    internal_min = min(internal_values)
    loose_threshold = max(0.58, config.tile_match_threshold - (0.14 if size >= 4 else 0.08 if size == 3 else 0.0))
    if internal_min < loose_threshold:
        return None
    internal_mean = float(np.mean(internal_values))
    if size >= 3 and internal_mean < config.tile_match_threshold:
        return None
    if size == 2 and internal_min < config.tile_match_threshold:
        return None

    outsider_values = []
    group_set = set(group_indices)
    for idx in group_indices:
        for other in neighbor_rankings[idx]:
            if other not in group_set:
                outsider_values.append(float(similarity_matrix[idx, other]))
                break
    outsider_best = max(outsider_values) if outsider_values else 0.0
    separation = internal_min - outsider_best

    member_support_values = []
    for idx in group_indices:
        neighbor_order = neighbor_rankings[idx]
        top_needed = max(size - 1, 1)
        top_neighbors = [neighbor for neighbor in neighbor_order if neighbor not in group_set][:top_needed]
        inside_neighbors = [neighbor for neighbor in neighbor_order if neighbor in group_set and neighbor != idx][:top_needed]
        support_hits = len(inside_neighbors)
        penalty_hits = len(top_neighbors)
        member_support = support_hits / max(top_needed, 1)
        if penalty_hits:
            penalty = max(0.0, 1.0 - np.mean([similarity_matrix[idx, neighbor] for neighbor in top_neighbors]))
            member_support = max(0.0, member_support - penalty * 0.15)
        member_support_values.append(member_support)
    member_support = float(np.mean(member_support_values))

    confidence = float(
        np.clip(
            0.45 * internal_mean
            + 0.30 * internal_min
            + 0.15 * member_support
            + 0.10 * max(separation, 0.0),
            0.0,
            1.0,
        )
    )
    return confidence, separation, internal_mean, member_support, internal_min


def infer_match_groups(
    features: dict[str, TileFeature],
    config: MatchTileConfig,
    observations: dict[str, CellObservation] | None = None,
) -> tuple[list[MatchGroup], list[str]]:
    cell_ids = list(features)
    if not cell_ids:
        return [], []

    max_group_size = max(2, min(config.max_group_size, 4, len(cell_ids)))
    primary_appearance = np.stack([features[cell_id].descriptor for cell_id in cell_ids], axis=0)
    appearance_matrix = np.clip(primary_appearance @ primary_appearance.T, 0.0, 1.0).astype(np.float32)
    if all(features[cell_id].shape_descriptor is not None for cell_id in cell_ids):
        primary_shape = np.stack([features[cell_id].shape_descriptor for cell_id in cell_ids], axis=0)
        shape_matrix = np.clip(primary_shape @ primary_shape.T, 0.0, 1.0).astype(np.float32)
    else:
        shape_matrix = appearance_matrix.copy()
    if all(features[cell_id].color_signature is not None for cell_id in cell_ids):
        primary_color = np.stack([features[cell_id].color_signature for cell_id in cell_ids], axis=0)
        color_matrix = np.clip(primary_color @ primary_color.T, 0.0, 1.0).astype(np.float32)
    else:
        color_matrix = appearance_matrix.copy()
    similarity_matrix = np.clip(0.20 * appearance_matrix + 0.80 * (shape_matrix * color_matrix), 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(similarity_matrix, -1.0)
    ranking_limit = min(len(cell_ids) - 1, max_group_size + 12)
    neighbor_rankings = [
        [int(idx) for idx in np.argsort(similarity_matrix[row])[::-1] if idx != row][:ranking_limit]
        for row in range(len(cell_ids))
    ]
    refinement_neighbors = min(len(cell_ids) - 1, max_group_size + 6)
    refined_pairs: set[tuple[int, int]] = set()
    for left in range(len(cell_ids)):
        neighbor_order = neighbor_rankings[left][:refinement_neighbors]
        for right in neighbor_order:
            pair = (left, right) if left < right else (right, left)
            if pair in refined_pairs:
                continue
            refined_pairs.add(pair)
            score = compare_features(features[cell_ids[pair[0]]], features[cell_ids[pair[1]]])
            similarity_matrix[pair[0], pair[1]] = score
            similarity_matrix[pair[1], pair[0]] = score
    neighbor_rankings = [
        [int(idx) for idx in np.argsort(similarity_matrix[row])[::-1] if idx != row][:ranking_limit]
        for row in range(len(cell_ids))
    ]
    neighbor_pool = min(len(cell_ids) - 1, max(max_group_size, 4))

    candidate_records: dict[frozenset[int], dict[str, float | tuple[int, ...]]] = {}
    for anchor in range(len(cell_ids)):
        neighbor_order = [int(idx) for idx in np.argsort(similarity_matrix[anchor])[::-1] if idx != anchor][:neighbor_pool]
        for size in range(2, max_group_size + 1):
            if len(neighbor_order) < size - 1:
                continue
            for combo in combinations(neighbor_order, size - 1):
                group_indices = tuple(sorted((anchor, *combo)))
                key = frozenset(group_indices)
                metrics = _candidate_confidence(similarity_matrix, group_indices, neighbor_rankings, config)
                if metrics is None:
                    continue
                confidence, separation, internal_mean, member_support, internal_min = metrics
                previous = candidate_records.get(key)
                if previous and float(previous["confidence"]) >= confidence:
                    continue
                candidate_records[key] = {
                    "indices": group_indices,
                    "confidence": confidence,
                    "separation": separation,
                    "internal_mean": internal_mean,
                    "member_support": member_support,
                    "internal_min": internal_min,
                }

    if not candidate_records:
        return [], sorted(cell_ids)

    candidates = list(candidate_records.values())
    overlap_margin = 0.02
    for candidate in candidates:
        indices = set(candidate["indices"])
        candidate["adjusted_score"] = float(candidate["confidence"]) + max(0, len(candidate["indices"]) - 2) * 0.10
        required_confidence = config.group_confidence_threshold - (0.04 if len(indices) >= 4 else 0.02 if len(indices) == 3 else 0.0)
        candidate["base_ambiguous"] = bool(
            float(candidate["confidence"]) < required_confidence
            or float(candidate["separation"]) < 0.03
            or float(candidate["member_support"]) < 0.70
        )
    for candidate in candidates:
        indices = set(candidate["indices"])
        competing = [
            float(other["adjusted_score"])
            for other in candidates
            if other is not candidate
            and indices.intersection(other["indices"])
            and not set(other["indices"]).issubset(indices)
        ]
        candidate["competing_score"] = max(competing) if competing else 0.0
        candidate["ambiguous"] = bool(
            bool(candidate["base_ambiguous"])
            or (competing and float(candidate["competing_score"]) >= float(candidate["adjusted_score"]) - overlap_margin)
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not bool(candidate["ambiguous"]),
            float(candidate["adjusted_score"]),
            len(candidate["indices"]),
            float(candidate["separation"]),
        ),
        reverse=True,
    )

    used: set[int] = set()
    groups: list[MatchGroup] = []
    label_number = 1
    for candidate in ordered:
        indices = tuple(int(idx) for idx in candidate["indices"])
        if any(idx in used for idx in indices):
            continue
        if bool(candidate["ambiguous"]):
            continue
        members = [cell_ids[idx] for idx in indices]
        click_order, click_order_positions = _ordered_group_members(members, observations or {})
        groups.append(
            MatchGroup(
                label=f"G{label_number:03d}",
                members=members,
                confidence=round(float(candidate["confidence"]), 3),
                ambiguous=bool(candidate["ambiguous"]),
                group_size=len(members),
                click_order=click_order,
                click_order_positions=click_order_positions,
                similarity_min=round(float(candidate["internal_min"]), 3),
                similarity_mean=round(float(candidate["internal_mean"]), 3),
            )
        )
        used.update(indices)
        label_number += 1

    best_scores = {
        cell_ids[index]: float(np.max(similarity_matrix[index]))
        for index in range(len(cell_ids))
    }
    unresolved = [cell_ids[index] for index in range(len(cell_ids)) if index not in used]
    unresolved.sort(key=lambda cell_id: best_scores.get(cell_id, 0.0), reverse=True)
    return groups, unresolved


def reconstruct_from_session(session_dir: Path, config: MatchTileConfig, calibration: Calibration) -> ReconstructionResult:
    frames_dir = session_dir / "frames"
    image_paths = sorted(frames_dir.glob("*.png"))
    result = reconstruct_from_images(image_paths, session_dir=session_dir, config=config, calibration=calibration)
    result.save(session_dir / "result.json")
    return result
