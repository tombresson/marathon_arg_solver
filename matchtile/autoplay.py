from __future__ import annotations

import time
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput.mouse import Button, Controller

from matchtile.models import MatchGroup, ReconstructionResult
from matchtile.runtime_control import StopToken
from matchtile.vision import cell_center, warp_image_to_board


def _cell_screen_point(result: ReconstructionResult, cell_id: str, screen_offset: tuple[int, int]) -> tuple[int, int]:
    obs = result.observations[cell_id]
    x, y = cell_center(result.calibration, obs.row, obs.col)
    x += screen_offset[0]
    y += screen_offset[1]
    return int(round(x)), int(round(y))


def _capture_cell(result: ReconstructionResult, cell_id: str, capture_rect: tuple[int, int, int, int], screen_offset: tuple[int, int]) -> np.ndarray | None:
    left, top, width, height = capture_rect
    with mss.mss() as sct:
        grabbed = sct.grab({"left": left, "top": top, "width": width, "height": height})
        frame = np.array(grabbed, dtype=np.uint8)[..., :3]

    board = warp_image_to_board(frame, result.calibration, image_origin=screen_offset)
    obs = result.observations[cell_id]
    x = int(obs.col * result.calibration.pitch_x)
    y = int(obs.row * result.calibration.pitch_y)
    w = int(result.calibration.pitch_x)
    h = int(result.calibration.pitch_y)
    crop = board[max(y, 0) : min(y + h, board.shape[0]), max(x, 0) : min(x + w, board.shape[1])]
    if crop.size == 0:
        return None
    return crop


def _verify_group_change(result: ReconstructionResult, group: MatchGroup, capture_rect: tuple[int, int, int, int], screen_offset: tuple[int, int]) -> bool:
    changed = 0
    for cell_id in group.members:
        current = _capture_cell(result, cell_id, capture_rect, screen_offset)
        obs = result.observations.get(cell_id)
        if current is None or not obs or not obs.crop_path:
            continue
        reference = cv2.imread(obs.crop_path, cv2.IMREAD_COLOR)
        if reference is None:
            continue
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
        delta = float(np.mean(np.abs(current.astype(np.float32) - reference.astype(np.float32))))
        if delta > 18.0:
            changed += 1
    return changed == len(group.members)


def auto_click_groups(
    result_path: Path,
    min_confidence: float,
    click_delay_s: float,
    verify: bool = True,
    include_ambiguous: bool = False,
    capture_rect: tuple[int, int, int, int] | None = None,
    screen_offset: tuple[int, int] = (0, 0),
    stop_token: StopToken | None = None,
) -> int:
    result = ReconstructionResult.load(result_path)
    controller = Controller()
    groups = [
        group
        for group in result.groups
        if group.confidence >= min_confidence and (include_ambiguous or not group.ambiguous)
    ]
    groups.sort(key=lambda group: (group.confidence, group.group_size), reverse=True)

    clicked = 0
    for group in groups:
        if stop_token:
            stop_token.checkpoint()
        for cell_id in group.members:
            point = _cell_screen_point(result, cell_id, screen_offset)
            controller.position = point
            if stop_token:
                stop_token.sleep(click_delay_s / 2)
            else:
                time.sleep(click_delay_s / 2)
            controller.click(Button.left, 1)
            clicked += 1
            if stop_token:
                stop_token.sleep(click_delay_s)
            else:
                time.sleep(click_delay_s)
        if stop_token:
            stop_token.sleep(click_delay_s)
        else:
            time.sleep(click_delay_s)
        if verify and capture_rect:
            if not _verify_group_change(result, group, capture_rect, screen_offset):
                raise RuntimeError(
                    f"Auto-click verification failed on {group.label} "
                    f"(size {group.group_size}); aborting further clicks."
                )
    return clicked


def auto_click_pairs(*args, **kwargs) -> int:
    return auto_click_groups(*args, **kwargs)
