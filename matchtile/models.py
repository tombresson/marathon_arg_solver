from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class CellCenter:
    row: int
    col: int
    x: float
    y: float


@dataclass(slots=True)
class Calibration:
    board_rect: Rect
    rows: int
    cols: int
    pitch_x: float
    pitch_y: float
    offset_x: float
    offset_y: float
    source: str = "auto"
    centers: list[CellCenter] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["board_rect"] = self.board_rect.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        centers = [CellCenter(**entry) for entry in data.get("centers", [])]
        rest = {k: v for k, v in data.items() if k not in {"board_rect", "centers"}}
        return cls(board_rect=Rect(**data["board_rect"]), centers=centers, **rest)


@dataclass(slots=True)
class CellObservation:
    row: int
    col: int
    frame_index: int
    reveal_score: float
    sharpness: float
    crop_path: str | None = None

    def cell_id(self) -> str:
        return f"r{self.row:02d}c{self.col:02d}"


@dataclass(slots=True)
class MatchGroup:
    label: str
    members: list[str]
    confidence: float
    ambiguous: bool = False
    group_size: int = 2


@dataclass(slots=True)
class ReconstructionResult:
    calibration: Calibration
    observations: dict[str, CellObservation]
    groups: list[MatchGroup]
    unresolved: list[str]
    session_dir: str
    grid_composed_path: str | None = None
    grid_debug_path: str | None = None
    grid_fit_debug_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "calibration": self.calibration.to_dict(),
            "observations": {key: asdict(value) for key, value in self.observations.items()},
            "groups": [asdict(group) for group in self.groups],
            "unresolved": self.unresolved,
            "session_dir": self.session_dir,
            "grid_composed_path": self.grid_composed_path,
            "grid_debug_path": self.grid_debug_path,
            "grid_fit_debug_path": self.grid_fit_debug_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReconstructionResult":
        raw_groups = data.get("groups")
        if raw_groups is None:
            raw_groups = data.get("pairs", [])
        return cls(
            calibration=Calibration.from_dict(data["calibration"]),
            observations={key: CellObservation(**value) for key, value in data["observations"].items()},
            groups=[
                MatchGroup(
                    label=group["label"],
                    members=group["members"],
                    confidence=group["confidence"],
                    ambiguous=group.get("ambiguous", False),
                    group_size=group.get("group_size", len(group["members"])),
                )
                for group in raw_groups
            ],
            unresolved=data["unresolved"],
            session_dir=data["session_dir"],
            grid_composed_path=data.get("grid_composed_path"),
            grid_debug_path=data.get("grid_debug_path"),
            grid_fit_debug_path=data.get("grid_fit_debug_path"),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReconstructionResult":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
