from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import win32con
import win32gui

from matchtile.config import MatchTileConfig
from matchtile.debug_grid import render_debug_grids
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
class TileFeature:
    cell_id: str
    descriptor: np.ndarray


def _build_feature(cell_id: str, crop: np.ndarray) -> TileFeature:
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
    return TileFeature(cell_id=cell_id, descriptor=descriptor)


def compare_features(left: TileFeature, right: TileFeature) -> float:
    return float(np.clip(np.dot(left.descriptor, right.descriptor), 0.0, 1.0))


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
    observations: dict[str, CellObservation] = {}
    features: dict[str, TileFeature] = {}

    crops_dir = session_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(images):
        current_board = warp_image_to_board(frame, calibration)

        sat_scores: list[float] = []
        val_scores: list[float] = []
        std_scores: list[float] = []
        sharp_scores: list[float] = []
        cell_crops: dict[tuple[int, int], np.ndarray] = {}
        for row in range(calibration.rows):
            for col in range(calibration.cols):
                crop = _cell_crop(current_board, calibration, row, col)
                if crop is None:
                    continue
                cell_crops[(row, col)] = crop
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sat_scores.append(float(hsv[..., 1].mean()))
                val_scores.append(float(hsv[..., 2].mean()))
                std_scores.append(float(gray.std()))
                sharp_scores.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))

        sat_floor = max(12.0, float(np.quantile(sat_scores, 0.80)) * 0.18) if sat_scores else 12.0
        val_floor = max(36.0, float(np.quantile(val_scores, 0.75)) * 0.45) if val_scores else 36.0
        std_floor = max(config.frame_stability_threshold, float(np.quantile(std_scores, 0.80)) * 0.30) if std_scores else config.frame_stability_threshold
        sharp_floor = max(40.0, float(np.quantile(sharp_scores, 0.70)) * 0.20) if sharp_scores else 40.0

        for (row, col), crop in cell_crops.items():
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sat_mean = float(hsv[..., 1].mean())
            val_mean = float(hsv[..., 2].mean())
            std_gray = float(gray.std())
            reveal_score, sharpness = _reveal_score(crop)
            cell_id = f"r{row:02d}c{col:02d}"
            current = observations.get(cell_id)
            enough_signal = sat_mean >= sat_floor and val_mean >= val_floor and std_gray >= std_floor and sharpness >= sharp_floor
            if not enough_signal:
                continue
            if current and (reveal_score, sharpness) <= (current.reveal_score, current.sharpness):
                continue
            crop_path = crops_dir / f"{cell_id}.png"
            cv2.imwrite(str(crop_path), crop)
            observations[cell_id] = CellObservation(
                row=row,
                col=col,
                frame_index=frame_index,
                reveal_score=round(reveal_score, 3),
                sharpness=round(sharpness, 3),
                crop_path=str(crop_path),
            )
            features[cell_id] = _build_feature(cell_id, crop)

    groups, unresolved = infer_match_groups(features, config)
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
    return result


def _candidate_confidence(
    similarity_matrix: np.ndarray,
    group_indices: tuple[int, ...],
    config: MatchTileConfig,
) -> tuple[float, float, float, float] | None:
    size = len(group_indices)
    internal_values = [
        float(similarity_matrix[left, right])
        for left, right in combinations(group_indices, 2)
    ]
    if not internal_values:
        return None
    internal_min = min(internal_values)
    if internal_min < config.tile_match_threshold:
        return None
    internal_mean = float(np.mean(internal_values))

    outsider_values = []
    group_set = set(group_indices)
    for idx in group_indices:
        outside = [float(similarity_matrix[idx, other]) for other in range(similarity_matrix.shape[0]) if other not in group_set]
        if outside:
            outsider_values.append(max(outside))
    outsider_best = max(outsider_values) if outsider_values else 0.0
    separation = internal_min - outsider_best

    member_support_values = []
    for idx in group_indices:
        neighbor_order = np.argsort(similarity_matrix[idx])[::-1]
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
    return confidence, separation, internal_mean, member_support


def infer_match_groups(features: dict[str, TileFeature], config: MatchTileConfig) -> tuple[list[MatchGroup], list[str]]:
    cell_ids = list(features)
    if not cell_ids:
        return [], []

    matrix = np.stack([features[cell_id].descriptor for cell_id in cell_ids], axis=0)
    similarity_matrix = np.clip(matrix @ matrix.T, 0.0, 1.0)
    np.fill_diagonal(similarity_matrix, -1.0)
    max_group_size = max(2, min(config.max_group_size, 4, len(cell_ids)))
    neighbor_pool = min(len(cell_ids) - 1, max_group_size + 2)

    candidate_records: dict[frozenset[int], dict[str, float | tuple[int, ...]]] = {}
    for anchor in range(len(cell_ids)):
        neighbor_order = [int(idx) for idx in np.argsort(similarity_matrix[anchor])[::-1] if idx != anchor][:neighbor_pool]
        for size in range(2, max_group_size + 1):
            if len(neighbor_order) < size - 1:
                continue
            for combo in combinations(neighbor_order, size - 1):
                group_indices = tuple(sorted((anchor, *combo)))
                key = frozenset(group_indices)
                metrics = _candidate_confidence(similarity_matrix, group_indices, config)
                if metrics is None:
                    continue
                confidence, separation, internal_mean, member_support = metrics
                previous = candidate_records.get(key)
                if previous and float(previous["confidence"]) >= confidence:
                    continue
                candidate_records[key] = {
                    "indices": group_indices,
                    "confidence": confidence,
                    "separation": separation,
                    "internal_mean": internal_mean,
                    "member_support": member_support,
                }

    if not candidate_records:
        return [], sorted(cell_ids)

    candidates = list(candidate_records.values())
    overlap_margin = 0.02
    for candidate in candidates:
        indices = set(candidate["indices"])
        competing = [
            float(other["confidence"])
            for other in candidates
            if other is not candidate and indices.intersection(other["indices"])
        ]
        candidate["competing_score"] = max(competing) if competing else 0.0
        candidate["ambiguous"] = bool(
            float(candidate["confidence"]) < config.group_confidence_threshold
            or float(candidate["separation"]) < 0.03
            or float(candidate["member_support"]) < 0.70
            or (competing and float(candidate["competing_score"]) >= float(candidate["confidence"]) - overlap_margin)
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not bool(candidate["ambiguous"]),
            float(candidate["confidence"]),
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
        if bool(candidate["ambiguous"]) and float(candidate["competing_score"]) >= float(candidate["confidence"]) - overlap_margin:
            continue
        members = [cell_ids[idx] for idx in indices]
        groups.append(
            MatchGroup(
                label=f"G{label_number:03d}",
                members=members,
                confidence=round(float(candidate["confidence"]), 3),
                ambiguous=bool(candidate["ambiguous"]),
                group_size=len(members),
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
