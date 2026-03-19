from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("matchtile.json")


@dataclass(slots=True)
class MatchTileConfig:
    window_title_regex: str = r"Discord|NULL//TRANSMIT\.ERR"
    capture_fps: int = 18
    reveal_duration_s: float = 10.0
    frame_stability_threshold: float = 16.0
    tile_match_threshold: float = 0.78
    pair_confidence_threshold: float = 0.84
    group_confidence_threshold: float = 0.84
    max_group_size: int = 4
    session_output_dir: str = "sessions"
    overlay_palette: list[str] = field(
        default_factory=lambda: [
            "#4CC9F0",
            "#F72585",
            "#B8F000",
            "#FF9F1C",
            "#E0FBFC",
            "#80ED99",
            "#C77DFF",
            "#F94144",
            "#FFD166",
            "#43AA8B",
        ]
    )

    @classmethod
    def load(cls, path: Path | None = None) -> "MatchTileConfig":
        config_path = path or DEFAULT_CONFIG_PATH
        if not config_path.exists():
            cfg = cls()
            cfg.save(config_path)
            return cfg
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: Path | None = None) -> None:
        config_path = path or DEFAULT_CONFIG_PATH
        config_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def session_dir(self) -> Path:
        path = Path(self.session_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
