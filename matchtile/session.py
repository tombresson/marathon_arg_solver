from __future__ import annotations

from datetime import datetime
from pathlib import Path

from matchtile.config import MatchTileConfig


def create_session_dir(config: MatchTileConfig) -> Path:
    root = config.session_dir()
    session_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "frames").mkdir(exist_ok=True)
    (session_dir / "crops").mkdir(exist_ok=True)
    return session_dir
