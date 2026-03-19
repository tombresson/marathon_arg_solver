from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from matchtile.config import MatchTileConfig
from matchtile.debug_grid import render_debug_grids
from matchtile.models import Calibration, CellCenter, CellObservation, MatchGroup, Rect, ReconstructionResult


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


def _artifact_mask(gray: np.ndarray) -> np.ndarray:
    bright_threshold = max(120, int(np.quantile(gray, 0.995)))
    mask = (gray >= bright_threshold).astype(np.uint8) * 255
    if mask.any():
        mask = cv2.dilate(mask, np.ones((11, 11), dtype=np.uint8), iterations=1)
    return mask


def _grid_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact_mask = _artifact_mask(gray)
    base = gray.astype(np.float32)
    high_pass = cv2.GaussianBlur(base, (0, 0), 1.2) - cv2.GaussianBlur(base, (0, 0), 6.0)
    high_pass = np.abs(high_pass)
    high_pass[artifact_mask > 0] = 0

    dots = cv2.GaussianBlur(base, (0, 0), 0.8) - cv2.GaussianBlur(base, (0, 0), 2.2)
    dots = np.maximum(dots, 0)
    dots[artifact_mask > 0] = 0

    grad_x = np.abs(cv2.Sobel(high_pass, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(high_pass, cv2.CV_32F, 0, 1, ksize=3))
    structure = np.minimum(cv2.blur(grad_x, (31, 31)), cv2.blur(grad_y, (31, 31))) + 0.6 * cv2.blur(dots, (17, 17))
    return structure, dots, artifact_mask


def _aggregate_frames(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0].copy()
    target_h, target_w = images[-1].shape[:2]
    normalized = [
        image if image.shape[:2] == (target_h, target_w) else cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        for image in images
    ]
    stack = np.stack(normalized, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _aggregate_grid_maps(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(images) == 1:
        gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        structure, dots, _ = _grid_maps(gray)
        return structure, dots
    target_h, target_w = images[-1].shape[:2]
    structures: list[np.ndarray] = []
    dots_list: list[np.ndarray] = []
    for image in images:
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        structure, dots, _ = _grid_maps(gray)
        structures.append(structure)
        dots_list.append(dots)
    return np.mean(structures, axis=0), np.mean(dots_list, axis=0)


def _span_mask(signal: np.ndarray, quantile: float, kernel_size: int) -> np.ndarray:
    threshold = float(np.quantile(signal, quantile))
    mask = (signal >= threshold).astype(np.uint8)
    kernel = np.ones((max(kernel_size, 3),), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.reshape(1, -1), cv2.MORPH_CLOSE, kernel.reshape(1, -1), iterations=1).reshape(-1)
    return mask


def _largest_span(mask: np.ndarray, minimum_length: int) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0, mask.shape[0] - 1
    if int(indices[-1] - indices[0] + 1) >= minimum_length:
        return int(indices[0]), int(indices[-1])
    start = int(indices[0])
    best_start = start
    best_end = start
    prev = start
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            if prev - start > best_end - best_start:
                best_start, best_end = start, prev
            start = idx
        prev = idx
    if prev - start > best_end - best_start:
        best_start, best_end = start, prev
    if best_end - best_start + 1 < minimum_length:
        return int(indices[0]), int(indices[-1])
    return best_start, best_end


def detect_board_rect_from_structure(structure: np.ndarray, restrict_left_bias: bool = False) -> Rect:
    height, width = structure.shape[:2]
    col_score = cv2.GaussianBlur(structure.mean(axis=0).reshape(1, -1), (0, 0), max(8.0, width / 180.0)).reshape(-1)
    row_score = cv2.GaussianBlur(structure.mean(axis=1).reshape(-1, 1), (0, 0), max(8.0, height / 180.0)).reshape(-1)
    if restrict_left_bias:
        col_score[: int(width * 0.06)] = 0
    col_mask = _span_mask(col_score, quantile=0.50, kernel_size=max(31, width // 50))
    row_mask = _span_mask(row_score, quantile=0.50, kernel_size=max(31, height // 35))
    x0, x1 = _largest_span(col_mask, minimum_length=max(width // 3, 200))
    y0, y1 = _largest_span(row_mask, minimum_length=max(height // 4, 160))
    pad_x = max(4, int(round((x1 - x0 + 1) * 0.002)))
    pad_y = max(4, int(round((y1 - y0 + 1) * 0.002)))
    x0 = max(x0 - pad_x, 0)
    y0 = max(y0 - pad_y, 0)
    x1 = min(x1 + pad_x, width - 1)
    y1 = min(y1 + pad_y, height - 1)
    return Rect(x0, y0, max(x1 - x0, 1), max(y1 - y0, 1))


def detect_board_rect(image: np.ndarray, restrict_left_bias: bool = False) -> Rect:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    structure, _, _ = _grid_maps(gray)
    return detect_board_rect_from_structure(structure, restrict_left_bias=restrict_left_bias)


def _dominant_period(projection: np.ndarray, min_period: int = 12, max_period: int = 40) -> int:
    signal = projection.astype(np.float32)
    signal = signal - signal.mean()
    best_period = min_period
    best_score = float("-inf")
    for period in range(min_period, max_period + 1):
        if period >= signal.shape[0]:
            break
        score = float(np.dot(signal[:-period], signal[period:]))
        if score > best_score:
            best_score = score
            best_period = period
    return best_period


def _best_phase(projection: np.ndarray, period: int) -> int:
    best_phase = 0
    best_score = float("-inf")
    for phase in range(period):
        score = float(projection[phase::period].sum())
        if score > best_score:
            best_score = score
            best_phase = phase
    return best_phase


def _period_scores(projection: np.ndarray, min_period: int, max_period: int) -> dict[int, float]:
    signal = projection.astype(np.float32)
    signal = signal - signal.mean()
    scores: dict[int, float] = {}
    for period in range(min_period, max_period + 1):
        if period >= signal.shape[0]:
            break
        scores[period] = float(np.dot(signal[:-period], signal[period:]))
    return scores


def _fundamental_period(projection: np.ndarray, min_period: int, max_period: int) -> float:
    scores = _period_scores(projection, min_period=min_period, max_period=max_period)
    if not scores:
        return float(min_period)
    best_period = max(scores, key=scores.get)
    harmonic_candidates = [period for period in (best_period * 2 - 1, best_period * 2, best_period * 2 + 1) if period in scores]
    if harmonic_candidates:
        harmonic_period = max(harmonic_candidates, key=lambda period: scores[period])
        if scores[harmonic_period] >= scores[best_period] * 0.78:
            return float(harmonic_period)
    return float(best_period)


def _sample_axis_profile(signal: np.ndarray, positions: np.ndarray, sigma: float) -> tuple[float, float]:
    values: list[float] = []
    for position in positions:
        low = max(int(np.floor(position - 3.0 * sigma)), 0)
        high = min(int(np.ceil(position + 3.0 * sigma)) + 1, signal.shape[0])
        if low >= high:
            continue
        coords = np.arange(low, high, dtype=np.float32)
        weights = np.exp(-0.5 * ((coords - position) / sigma) ** 2)
        values.append(float((signal[low:high] * weights).sum() / max(weights.sum(), 1e-6)))
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.quantile(values, 0.25))


def _score_axis_count(
    dots_map: np.ndarray,
    line_projection: np.ndarray,
    count: int,
    axis: str,
    seed_pitches: tuple[float, ...],
) -> tuple[float, float]:
    length = dots_map.shape[1] if axis == "x" else dots_map.shape[0]
    pitch = length / max(count, 1)
    sigma = max(1.0, pitch * 0.12)
    centers = (np.arange(count, dtype=np.float32) + 0.5) * pitch
    boundaries = np.arange(count + 1, dtype=np.float32) * pitch

    center_scores: list[float] = []
    samples = np.linspace(0.15, 0.85, 8)
    if axis == "x":
        for fraction in samples:
            row_index = int(round(fraction * (dots_map.shape[0] - 1)))
            mean_value, lower_quartile = _sample_axis_profile(dots_map[row_index], centers, sigma)
            center_scores.append(0.7 * mean_value + 0.3 * lower_quartile)
    else:
        for fraction in samples:
            col_index = int(round(fraction * (dots_map.shape[1] - 1)))
            mean_value, lower_quartile = _sample_axis_profile(dots_map[:, col_index], centers, sigma)
            center_scores.append(0.7 * mean_value + 0.3 * lower_quartile)

    line_mean, line_lower_quartile = _sample_axis_profile(line_projection, boundaries, sigma)
    center_support = float(np.mean(center_scores)) if center_scores else 0.0
    line_support = 0.7 * line_mean + 0.3 * line_lower_quartile
    seed_support = max(
        float(np.exp(-abs(pitch - seed_pitch) / max(2.0, seed_pitch * 0.20)))
        for seed_pitch in seed_pitches
        if seed_pitch > 0
    )
    return 0.55 * center_support + 0.25 * line_support + 0.20 * seed_support, pitch


def _choose_grid_counts_from_maps(structure: np.ndarray, dots: np.ndarray) -> tuple[int, int, float, float]:
    dots_map = cv2.GaussianBlur(dots, (0, 0), 1.2)
    line_x = cv2.GaussianBlur(structure.mean(axis=0).reshape(1, -1), (0, 0), 2.0).reshape(-1)
    line_y = cv2.GaussianBlur(structure.mean(axis=1).reshape(-1, 1), (0, 0), 2.0).reshape(-1)

    dominant_x = _fundamental_period(line_x, min_period=max(12, dots_map.shape[1] // 180), max_period=min(48, max(18, dots_map.shape[1] // 8)))
    dominant_y = _fundamental_period(line_y, min_period=max(12, dots_map.shape[0] // 90), max_period=min(48, max(18, dots_map.shape[0] // 4)))
    dense_seed_pitch_x = dominant_x * 0.90
    dense_seed_pitch_y = dominant_y * 0.90
    dense_seed_cols = max(2, int(round(dots_map.shape[1] / max(dense_seed_pitch_x, 1.0))))
    dense_seed_rows = max(2, int(round(dots_map.shape[0] / max(dense_seed_pitch_y, 1.0))))

    col_candidates = []
    min_cols = max(2, dense_seed_cols - 3)
    max_cols = dense_seed_cols + 3
    for cols in range(min_cols, max_cols + 1):
        score, pitch = _score_axis_count(
            dots_map,
            line_x,
            cols,
            axis="x",
            seed_pitches=(dense_seed_pitch_x,),
        )
        col_candidates.append((cols, score, pitch))

    row_candidates = []
    min_rows = max(2, dense_seed_rows - 3)
    max_rows = dense_seed_rows + 3
    for rows in range(min_rows, max_rows + 1):
        score, pitch = _score_axis_count(
            dots_map,
            line_y,
            rows,
            axis="y",
            seed_pitches=(dense_seed_pitch_y,),
        )
        row_candidates.append((rows, score, pitch))

    best_choice = None
    best_score = float("-inf")
    best_seed_distance = float("inf")
    for cols, col_score, pitch_x in col_candidates:
        for rows, row_score, pitch_y in row_candidates:
            pitch_similarity = float(np.exp(-abs(pitch_x - pitch_y) / 4.0))
            seed_distance = abs(cols - dense_seed_cols) + abs(rows - dense_seed_rows)
            score = 0.5 * (col_score + row_score) + 0.25 * pitch_similarity - 0.03 * seed_distance
            seed_distance = abs(cols - dense_seed_cols) + abs(rows - dense_seed_rows)
            if score > best_score + 0.01 or (abs(score - best_score) <= 0.01 and seed_distance < best_seed_distance):
                best_score = score
                best_seed_distance = seed_distance
                best_choice = (rows, cols, pitch_y, pitch_x)
    if best_choice is None:
        return dense_seed_rows, dense_seed_cols, dots_map.shape[0] / max(dense_seed_rows, 1), dots_map.shape[1] / max(dense_seed_cols, 1)
    rows, cols, pitch_y, pitch_x = best_choice
    return int(rows), int(cols), float(pitch_y), float(pitch_x)


def _build_calibration(board_rect: Rect, rows: int, cols: int, pitch_y: float, pitch_x: float, source: str = "auto") -> Calibration:
    offset_x = 0.0
    offset_y = 0.0

    centers: list[CellCenter] = []
    for row in range(rows):
        for col in range(cols):
            center_x = board_rect.x + offset_x + (col + 0.5) * pitch_x
            center_y = board_rect.y + offset_y + (row + 0.5) * pitch_y
            if center_x >= board_rect.right or center_y >= board_rect.bottom:
                continue
            centers.append(CellCenter(row=row, col=col, x=center_x, y=center_y))

    return Calibration(
        board_rect=board_rect,
        rows=rows,
        cols=cols,
        pitch_x=float(pitch_x),
        pitch_y=float(pitch_y),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        source=source,
        centers=centers,
    )


def _choose_grid_counts(board_image: np.ndarray) -> tuple[int, int, float, float]:
    gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
    structure, dots, _ = _grid_maps(gray)
    return _choose_grid_counts_from_maps(structure, dots)


def calibrate_grid(board_image: np.ndarray, board_rect: Rect, source: str = "auto") -> Calibration:
    rows, cols, pitch_y, pitch_x = _choose_grid_counts(board_image)
    return _build_calibration(board_rect, rows, cols, pitch_y, pitch_x, source=source)


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


def _select_calibration_images(images: list[np.ndarray]) -> list[np.ndarray]:
    if len(images) <= 12:
        return images
    return images[-12:]


def render_grid_fit_debug(image: np.ndarray, calibration: Calibration, output_path: Path) -> Path:
    overlay = image.copy()
    board = calibration.board_rect
    cv2.rectangle(overlay, (board.x, board.y), (board.right, board.bottom), (0, 255, 0), 2)

    for col in range(calibration.cols + 1):
        x = int(round(board.x + col * calibration.pitch_x))
        cv2.line(overlay, (x, board.y), (x, board.bottom), (0, 255, 0), 1)
    for row in range(calibration.rows + 1):
        y = int(round(board.y + row * calibration.pitch_y))
        cv2.line(overlay, (board.x, y), (board.right, y), (0, 255, 0), 1)

    label = (
        f"{calibration.cols} cols x {calibration.rows} rows | "
        f"pitch {calibration.pitch_x:.2f} x {calibration.pitch_y:.2f}"
    )
    cv2.putText(overlay, label, (board.x, max(24, board.y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), overlay)
    return output_path


def reconstruct_from_images(image_paths: list[Path], session_dir: Path, config: MatchTileConfig, board_rect_override: Rect | None = None) -> ReconstructionResult:
    images = [load_image(path) for path in image_paths]
    if not images:
        raise ValueError("No images provided for reconstruction.")

    calibration_images = _select_calibration_images(images)
    calibration_image = calibration_images[-1]
    if board_rect_override:
        board_rect = board_rect_override
        calibration = calibrate_grid(crop_rect(calibration_image, board_rect), board_rect)
    else:
        aggregate_structure, aggregate_dots = _aggregate_grid_maps(calibration_images)
        activity_rect = detect_discord_activity_region(calibration_image)
        activity_structure = aggregate_structure[activity_rect.y : activity_rect.bottom, activity_rect.x : activity_rect.right]
        activity_dots = aggregate_dots[activity_rect.y : activity_rect.bottom, activity_rect.x : activity_rect.right]
        board_local_rect = detect_board_rect_from_structure(activity_structure, restrict_left_bias=False)
        board_rect = Rect(activity_rect.x + board_local_rect.x, activity_rect.y + board_local_rect.y, board_local_rect.width, board_local_rect.height)
        board_structure = activity_structure[board_local_rect.y : board_local_rect.bottom, board_local_rect.x : board_local_rect.right]
        board_dots = activity_dots[board_local_rect.y : board_local_rect.bottom, board_local_rect.x : board_local_rect.right]
        rows, cols, pitch_y, pitch_x = _choose_grid_counts_from_maps(board_structure, board_dots)
        calibration = _build_calibration(board_rect, rows, cols, pitch_y, pitch_x)
    observations: dict[str, CellObservation] = {}
    features: dict[str, TileFeature] = {}

    crops_dir = session_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame in enumerate(images):
        current_board = crop_rect(frame, board_rect)
        if current_board.shape[:2] != (board_rect.height, board_rect.width):
            current_board = cv2.resize(current_board, (board_rect.width, board_rect.height), interpolation=cv2.INTER_AREA)

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


def reconstruct_from_session(session_dir: Path, config: MatchTileConfig, board_rect_override: Rect | None = None) -> ReconstructionResult:
    frames_dir = session_dir / "frames"
    image_paths = sorted(frames_dir.glob("*.png"))
    result = reconstruct_from_images(image_paths, session_dir=session_dir, config=config, board_rect_override=board_rect_override)
    result.save(session_dir / "result.json")
    return result
