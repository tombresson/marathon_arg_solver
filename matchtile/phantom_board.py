from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

from matchtile.debug_grid import render_debug_grids
from matchtile.models import Calibration, CellObservation, MatchGroup, PhantomBoard, ReconstructionResult
from matchtile.runtime_control import StopToken
from matchtile.vision import build_manual_calibration, render_grid_fit_debug


PHANTOM_BOARD_URL = "https://marathon.winnower.garden/api/cryo-phantom-board"


def _request_json(url: str, timeout_s: float = 10.0) -> dict:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "matchtile/1.0",
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body)


def parse_phantom_metadata(payload: dict) -> dict:
    try:
        width = int(payload["width"])
        height = int(payload["height"])
        pair_size = int(payload["pairSize"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed phantom board metadata: {exc}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("Phantom board metadata dimensions must be positive.")
    if not 2 <= pair_size <= 4:
        raise RuntimeError(f"Unsupported phantom board pair size: {pair_size}.")
    return {
        "width": width,
        "height": height,
        "pair_size": pair_size,
        "started_at": payload.get("startedAt"),
        "timestamp": payload.get("timestamp"),
        "reported_at": payload.get("reportedAt"),
        "remaining_time": payload.get("remainingTime"),
    }


def fetch_phantom_metadata(url: str = PHANTOM_BOARD_URL, timeout_s: float = 10.0) -> dict:
    try:
        payload = _request_json(url, timeout_s=timeout_s)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to fetch phantom metadata from {url}: {exc}") from exc
    return parse_phantom_metadata(payload)


def save_phantom_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_phantom_metadata(path: Path) -> dict:
    return parse_phantom_metadata(json.loads(path.read_text(encoding="utf-8")))


def parse_phantom_board(payload: dict) -> PhantomBoard:
    try:
        board = PhantomBoard.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed phantom board payload: {exc}") from exc

    if board.width <= 0 or board.height <= 0:
        raise RuntimeError("Phantom board dimensions must be positive.")
    if not 2 <= board.pairSize <= 4:
        raise RuntimeError(f"Unsupported phantom board pair size: {board.pairSize}.")
    expected_cells = board.width * board.height
    if len(board.grid) != expected_cells:
        raise RuntimeError(f"Phantom board expected {expected_cells} cells but received {len(board.grid)}.")

    seen_positions: set[tuple[int, int]] = set()
    counts: Counter[int] = Counter()
    for cell in board.grid:
        if not 0 <= cell.row < board.height or not 0 <= cell.col < board.width:
            raise RuntimeError(f"Cell ({cell.row}, {cell.col}) is outside {board.width}x{board.height}.")
        position = (cell.row, cell.col)
        if position in seen_positions:
            raise RuntimeError(f"Duplicate phantom board position at row {cell.row}, col {cell.col}.")
        seen_positions.add(position)
        counts[cell.imgIdx] += 1
    invalid_counts = {img_idx: count for img_idx, count in counts.items() if count != board.pairSize}
    if invalid_counts:
        details = ", ".join(f"{img_idx}:{count}" for img_idx, count in sorted(invalid_counts.items())[:8])
        raise RuntimeError(
            "Phantom board symbol counts did not match pairSize "
            f"{board.pairSize}. Offending counts: {details}"
        )
    return board


def fetch_phantom_board(url: str = PHANTOM_BOARD_URL, timeout_s: float = 10.0) -> PhantomBoard:
    try:
        payload = _request_json(url, timeout_s=timeout_s)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to fetch phantom board from {url}: {exc}") from exc
    return parse_phantom_board(payload)


def phantom_board_hash(board: PhantomBoard) -> str:
    canonical = {
        "width": board.width,
        "height": board.height,
        "pairSize": board.pairSize,
        "grid": [
            {
                "row": cell.row,
                "col": cell.col,
                "imgIdx": cell.imgIdx,
                "url": cell.url,
            }
            for cell in sorted(board.grid, key=lambda item: (item.row, item.col))
        ],
    }
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest


def wait_for_fresh_phantom_board(
    baseline: PhantomBoard | None,
    url: str = PHANTOM_BOARD_URL,
    poll_interval_s: float = 0.25,
    timeout_s: float = 10.0,
    stop_token: StopToken | None = None,
) -> PhantomBoard:
    baseline_hash = phantom_board_hash(baseline) if baseline else None
    deadline = time.perf_counter() + timeout_s
    last_error: RuntimeError | None = None
    while time.perf_counter() < deadline:
        if stop_token:
            stop_token.checkpoint()
        try:
            candidate = fetch_phantom_board(url=url, timeout_s=min(5.0, timeout_s))
        except RuntimeError as exc:
            last_error = exc
        else:
            candidate_hash = phantom_board_hash(candidate)
            if baseline is None:
                return candidate
            started_changed = candidate.startedAt and candidate.startedAt != baseline.startedAt
            metadata_changed = (
                candidate.timestamp != baseline.timestamp
                or candidate.reportedAt != baseline.reportedAt
            )
            if started_changed or (metadata_changed and candidate_hash != baseline_hash):
                return candidate
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        sleep_for = min(poll_interval_s, remaining)
        if stop_token:
            stop_token.sleep(sleep_for)
        else:
            time.sleep(sleep_for)
    if last_error:
        raise RuntimeError(f"Timed out waiting for a fresh phantom board. Last error: {last_error}")
    raise RuntimeError("Timed out waiting for a fresh phantom board update after F8.")


def save_phantom_board(path: Path, board: PhantomBoard) -> None:
    path.write_text(json.dumps(board.to_dict(), indent=2), encoding="utf-8")


def load_phantom_board(path: Path) -> PhantomBoard:
    return parse_phantom_board(json.loads(path.read_text(encoding="utf-8")))


def derive_board_calibration(base_calibration: Calibration, rows: int, cols: int) -> Calibration:
    calibration = build_manual_calibration(
        (base_calibration.client_width, base_calibration.client_height),
        rows=rows,
        cols=cols,
        corners=base_calibration.board_corners,
        group_size=base_calibration.group_size,
        source="website-metadata",
    )
    polygon = np.array([[point.x, point.y] for point in calibration.board_corners], dtype=np.float32)
    for center in calibration.centers:
        inside = cv2.pointPolygonTest(polygon, (center.x, center.y), False)
        if inside < 0:
            raise RuntimeError(
                f"Derived board center r{center.row:02d}c{center.col:02d} fell outside the calibrated playfield. "
                "Re-run `matchtile calibrate` for this window size."
            )
    return calibration


def _tile_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return suffix
    return ".png"


def _placeholder_tile(path: Path, symbol_id: int, size: int = 72) -> str:
    tile = np.full((size, size, 3), 22, dtype=np.uint8)
    cv2.rectangle(tile, (1, 1), (size - 2, size - 2), (60, 60, 60), 1)
    cv2.rectangle(tile, (4, 4), (size - 5, size - 5), (36, 36, 36), 1)
    cv2.putText(
        tile,
        f"{symbol_id}",
        (8, size // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), tile)
    return str(path)


def _download_tile(url: str, out_path: Path, timeout_s: float = 10.0) -> str:
    request = Request(url, headers={"User-Agent": "matchtile/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        data = response.read()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Downloaded tile could not be decoded: {url}")
    cv2.imwrite(str(out_path), image)
    return str(out_path)


def ensure_board_tiles(board: PhantomBoard, tiles_dir: Path) -> dict[int, str]:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: dict[int, str] = {}
    by_symbol: dict[int, str] = {}
    for cell in board.grid:
        by_symbol.setdefault(cell.imgIdx, cell.url)

    def task(item: tuple[int, str]) -> tuple[int, str]:
        img_idx, url = item
        out_path = tiles_dir / f"img_{img_idx:04d}{_tile_extension(url)}"
        try:
            return img_idx, _download_tile(url, out_path)
        except Exception:
            placeholder_path = tiles_dir / f"img_{img_idx:04d}_placeholder.png"
            return img_idx, _placeholder_tile(placeholder_path, img_idx)

    max_workers = min(24, max(4, len(by_symbol)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, item) for item in sorted(by_symbol.items())]
        for future in as_completed(futures):
            img_idx, path = future.result()
            tile_paths[img_idx] = path
    return tile_paths


def _cell_id(row: int, col: int) -> str:
    return f"r{row:02d}c{col:02d}"


def _ordered_positions(cells: list[tuple[int, int]]) -> tuple[list[str], list[dict]]:
    ordered = sorted(cells, key=lambda item: (item[0], item[1]))
    return (
        [_cell_id(row, col) for row, col in ordered],
        [{"row": row, "col": col} for row, col in ordered],
    )


def build_reconstruction_from_phantom_board(
    board: PhantomBoard,
    base_calibration: Calibration,
    session_dir: Path,
    board_source: dict | None = None,
) -> ReconstructionResult:
    calibration = derive_board_calibration(base_calibration, rows=board.height, cols=board.width)
    tile_paths = ensure_board_tiles(board, session_dir / "tiles")
    observations: dict[str, CellObservation] = {}
    groups_by_symbol: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for cell in sorted(board.grid, key=lambda item: (item.row, item.col)):
        cell_id = _cell_id(cell.row, cell.col)
        observations[cell_id] = CellObservation(
            row=cell.row,
            col=cell.col,
            frame_index=-1,
            reveal_score=1.0,
            sharpness=1.0,
            crop_path=tile_paths.get(cell.imgIdx),
            state="website-board",
            quality_score=1.0,
            visible_width_ratio=1.0,
            occupancy_ratio=1.0,
            symbol_id=cell.imgIdx,
            image_url=cell.url,
            alternate_candidates=[],
            discarded_candidates=[],
            timeline=[],
        )
        groups_by_symbol[cell.imgIdx].append((cell.row, cell.col))

    ordered_groups: list[tuple[int, list[tuple[int, int]]]] = sorted(
        groups_by_symbol.items(),
        key=lambda item: min(item[1], key=lambda pos: (pos[0], pos[1])),
    )
    groups: list[MatchGroup] = []
    for index, (img_idx, cells) in enumerate(ordered_groups, start=1):
        click_order, click_order_positions = _ordered_positions(cells)
        groups.append(
            MatchGroup(
                label=f"G{index:03d}",
                members=click_order.copy(),
                confidence=1.0,
                ambiguous=False,
                group_size=len(cells),
                click_order=click_order,
                click_order_positions=click_order_positions,
                similarity_min=1.0,
                similarity_mean=1.0,
            )
        )

    result = ReconstructionResult(
        calibration=calibration,
        observations=observations,
        groups=groups,
        unresolved=[],
        session_dir=str(session_dir),
        board_source=board_source,
    )
    blank = np.full((calibration.client_height, calibration.client_width, 3), 12, dtype=np.uint8)
    fit_debug_path = render_grid_fit_debug(blank, calibration, session_dir / "grid_fit_debug.png")
    result.grid_fit_debug_path = str(fit_debug_path)
    composed_path, debug_path = render_debug_grids(result, session_dir)
    result.grid_composed_path = str(composed_path)
    result.grid_debug_path = str(debug_path)
    return result


def board_source_metadata(board: PhantomBoard, source_url: str, mode: str, fetched_at: datetime | None = None) -> dict:
    fetched = fetched_at or datetime.now(timezone.utc)
    return {
        "mode": mode,
        "url": source_url,
        "fetched_at": fetched.isoformat(),
        "board_hash": phantom_board_hash(board),
        "started_at": board.startedAt,
        "timestamp": board.timestamp,
        "reported_at": board.reportedAt,
        "remaining_time": board.remainingTime,
        "width": board.width,
        "height": board.height,
        "pair_size": board.pairSize,
    }


def metadata_source(metadata: dict, source_url: str, mode: str, fetched_at: datetime | None = None) -> dict:
    fetched = fetched_at or datetime.now(timezone.utc)
    return {
        "mode": mode,
        "url": source_url,
        "fetched_at": fetched.isoformat(),
        "started_at": metadata.get("started_at"),
        "timestamp": metadata.get("timestamp"),
        "reported_at": metadata.get("reported_at"),
        "remaining_time": metadata.get("remaining_time"),
        "width": metadata["width"],
        "height": metadata["height"],
        "pair_size": metadata["pair_size"],
    }
