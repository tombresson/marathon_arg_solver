from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from matchtile.models import ReconstructionResult


def _tile_size(result: ReconstructionResult) -> tuple[int, int]:
    width = max(20, int(round(result.calibration.pitch_x)))
    height = max(20, int(round(result.calibration.pitch_y)))
    return width, height


def _placeholder_tile(width: int, height: int, row: int, col: int) -> np.ndarray:
    tile = np.full((height, width, 3), 18, dtype=np.uint8)
    inner = (28 + ((row + col) % 2) * 6)
    cv2.rectangle(tile, (1, 1), (width - 2, height - 2), (40, 40, 40), 1)
    cv2.rectangle(tile, (3, 3), (width - 4, height - 4), (inner, inner, inner), 1)
    cv2.line(tile, (width // 4, height // 2), (3 * width // 4, height // 2), (55, 55, 55), 1)
    cv2.line(tile, (width // 2, height // 4), (width // 2, 3 * height // 4), (55, 55, 55), 1)
    return tile


def _load_or_placeholder(result: ReconstructionResult, cell_id: str, width: int, height: int, row: int, col: int) -> np.ndarray:
    obs = result.observations.get(cell_id)
    if obs and obs.crop_path:
        crop = cv2.imread(obs.crop_path, cv2.IMREAD_COLOR)
        if crop is not None:
            return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return _placeholder_tile(width, height, row, col)


def _group_lookup(result: ReconstructionResult) -> dict[str, tuple[str, float, bool, int]]:
    lookup: dict[str, tuple[str, float, bool, int]] = {}
    for group in result.groups:
        for cell_id in group.members:
            lookup[cell_id] = (group.label, group.confidence, group.ambiguous, group.group_size)
    return lookup


def _load_candidate_tile(candidate: dict | None, width: int, height: int, row: int, col: int) -> np.ndarray:
    if candidate:
        crop_path = candidate.get("crop_path")
        if crop_path:
            crop = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if crop is not None:
                return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    return _placeholder_tile(width, height, row, col)


def render_debug_grids(result: ReconstructionResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = _tile_size(result)
    rows = result.calibration.rows
    cols = result.calibration.cols
    canvas_h = rows * tile_h
    canvas_w = cols * tile_w

    composed = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    annotated = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    group_lookup = _group_lookup(result)
    unresolved = set(result.unresolved)

    for row in range(rows):
        for col in range(cols):
            cell_id = f"r{row:02d}c{col:02d}"
            x0 = col * tile_w
            y0 = row * tile_h
            x1 = x0 + tile_w
            y1 = y0 + tile_h
            tile = _load_or_placeholder(result, cell_id, tile_w, tile_h, row, col)
            composed[y0:y1, x0:x1] = tile

            annotated_tile = tile.copy()
            group_info = group_lookup.get(cell_id)
            if group_info:
                label, confidence, ambiguous, group_size = group_info
                color = (0, 200, 0) if not ambiguous else (0, 165, 255)
                alpha = 0.22 if not ambiguous else 0.36
                overlay = annotated_tile.copy()
                cv2.rectangle(overlay, (0, 0), (tile_w - 1, tile_h - 1), color, -1)
                annotated_tile = cv2.addWeighted(overlay, alpha, annotated_tile, 1.0 - alpha, 0.0)
                cv2.rectangle(annotated_tile, (1, 1), (tile_w - 2, tile_h - 2), color, 2)
                cv2.putText(
                    annotated_tile,
                    f"{label[1:]}/{group_size}",
                    (3, max(12, tile_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_tile,
                    f"{confidence:.2f}",
                    (3, tile_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            elif cell_id in unresolved:
                overlay = annotated_tile.copy()
                cv2.rectangle(overlay, (0, 0), (tile_w - 1, tile_h - 1), (0, 0, 180), -1)
                annotated_tile = cv2.addWeighted(overlay, 0.25, annotated_tile, 0.75, 0.0)
                cv2.rectangle(annotated_tile, (1, 1), (tile_w - 2, tile_h - 2), (0, 0, 255), 2)
                cv2.putText(
                    annotated_tile,
                    "?",
                    (tile_w // 2 - 4, max(12, tile_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                annotated_tile,
                cell_id,
                (2, min(tile_h - 2, 11)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (215, 215, 215),
                1,
                cv2.LINE_AA,
            )
            annotated[y0:y1, x0:x1] = annotated_tile

    composed_path = output_dir / "grid_composed.png"
    debug_path = output_dir / "grid_debug.png"
    cv2.imwrite(str(composed_path), composed)
    cv2.imwrite(str(debug_path), annotated)
    return composed_path, debug_path


def render_candidate_debug(result: ReconstructionResult, output_dir: Path, max_cells: int = 18) -> Path | None:
    interesting: list[tuple[str, object]] = []
    for cell_id, observation in result.observations.items():
        if observation.alternate_candidates or observation.discarded_candidates or observation.state != "fully-revealed":
            interesting.append((cell_id, observation))
    if not interesting:
        return None

    interesting.sort(
        key=lambda item: (
            item[1].state != "fully-revealed",
            len(item[1].discarded_candidates),
            len(item[1].alternate_candidates),
            item[1].quality_score,
        ),
        reverse=True,
    )
    interesting = interesting[:max_cells]

    tile_w, tile_h = _tile_size(result)
    label_h = 40
    cell_w = tile_w * 5 + 24
    rows = len(interesting)
    canvas = np.full((rows * (tile_h + label_h), cell_w, 3), 14, dtype=np.uint8)

    for row_index, (cell_id, observation) in enumerate(interesting):
        y0 = row_index * (tile_h + label_h)
        cv2.putText(
            canvas,
            f"{cell_id} | {observation.state} | q={observation.quality_score:.2f}",
            (8, y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "chosen         alt1           alt2           drop1          drop2",
            (8, y0 + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 200, 255),
            1,
            cv2.LINE_AA,
        )
        slots = [
            {
                "crop_path": observation.crop_path,
                "state": observation.state,
                "frame_index": observation.frame_index,
                "quality_score": observation.quality_score,
            },
            *observation.alternate_candidates[:2],
            *observation.discarded_candidates[:2],
        ]
        while len(slots) < 5:
            slots.append(None)
        for slot_index, candidate in enumerate(slots):
            x0 = 8 + slot_index * tile_w
            tile = _load_candidate_tile(candidate, tile_w - 2, tile_h - 2, observation.row, observation.col)
            canvas[y0 + label_h : y0 + label_h + tile_h - 2, x0 : x0 + tile_w - 2] = tile
            cv2.rectangle(canvas, (x0, y0 + label_h), (x0 + tile_w - 2, y0 + label_h + tile_h - 2), (60, 60, 60), 1)
            if candidate:
                label = f"f{candidate.get('frame_index', -1)} {candidate.get('state', '?')[:6]}"
                cv2.putText(
                    canvas,
                    label,
                    (x0 + 1, y0 + label_h + tile_h + 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )

    output_path = output_dir / "candidate_debug.png"
    cv2.imwrite(str(output_path), canvas)
    return output_path
