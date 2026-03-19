from __future__ import annotations

import re
from pathlib import Path

from matchtile.config import MatchTileConfig
from matchtile.models import Calibration


def _safe_title(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "window"


def calibration_dir(config: MatchTileConfig) -> Path:
    path = Path(config.calibration_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def calibration_profile_path(config: MatchTileConfig, window_title: str, client_width: int, client_height: int) -> Path:
    return calibration_dir(config) / f"{_safe_title(window_title)}_{client_width}x{client_height}.calibration.json"


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(calibration.to_dict(), indent=2), encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    return Calibration.from_dict(__import__("json").loads(path.read_text(encoding="utf-8")))

