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
class Point:
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
    group_size: int = 2
    board_corners: list[Point] = field(default_factory=list)
    anchor_rows: int = 0
    anchor_cols: int = 0
    anchor_pitch_x: float = 0.0
    anchor_pitch_y: float = 0.0
    anchor_corners: list[Point] = field(default_factory=list)
    client_width: int = 0
    client_height: int = 0
    rectified_width: int = 0
    rectified_height: int = 0
    source: str = "manual-corners"
    centers: list[CellCenter] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["board_rect"] = self.board_rect.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        centers = [CellCenter(**entry) for entry in data.get("centers", [])]
        raw_corners = data.get("board_corners")
        if raw_corners:
            board_corners = [Point(**entry) for entry in raw_corners]
        else:
            rect = Rect(**data["board_rect"])
            board_corners = [
                Point(float(rect.x), float(rect.y)),
                Point(float(rect.right), float(rect.y)),
                Point(float(rect.right), float(rect.bottom)),
                Point(float(rect.x), float(rect.bottom)),
            ]
        raw_anchor_corners = data.get("anchor_corners")
        if raw_anchor_corners:
            anchor_corners = [Point(**entry) for entry in raw_anchor_corners]
        else:
            anchor_corners = [Point(point.x, point.y) for point in board_corners]
        rest = {k: v for k, v in data.items() if k not in {"board_rect", "centers", "board_corners", "anchor_corners"}}
        rest.setdefault("group_size", 2)
        rest.setdefault("anchor_rows", rest.get("rows", 0))
        rest.setdefault("anchor_cols", rest.get("cols", 0))
        rest.setdefault("anchor_pitch_x", rest.get("pitch_x", 0.0))
        rest.setdefault("anchor_pitch_y", rest.get("pitch_y", 0.0))
        rest.setdefault("client_width", data["board_rect"]["width"])
        rest.setdefault("client_height", data["board_rect"]["height"])
        rest.setdefault("rectified_width", max(int(round(rest.get("pitch_x", 0.0) * rest.get("cols", 0))), 0))
        rest.setdefault("rectified_height", max(int(round(rest.get("pitch_y", 0.0) * rest.get("rows", 0))), 0))
        return cls(
            board_rect=Rect(**data["board_rect"]),
            board_corners=board_corners,
            anchor_corners=anchor_corners,
            centers=centers,
            **rest,
        )


@dataclass(slots=True)
class PhantomCell:
    row: int
    col: int
    imgIdx: int
    url: str


@dataclass(slots=True)
class PhantomBoard:
    width: int
    height: int
    pairSize: int
    startedAt: str | None
    timestamp: str | None
    reportedAt: int | None
    remainingTime: int | None
    grid: list[PhantomCell]

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "pairSize": self.pairSize,
            "startedAt": self.startedAt,
            "timestamp": self.timestamp,
            "reportedAt": self.reportedAt,
            "remainingTime": self.remainingTime,
            "grid": [asdict(cell) for cell in self.grid],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhantomBoard":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            pairSize=int(data["pairSize"]),
            startedAt=data.get("startedAt"),
            timestamp=data.get("timestamp"),
            reportedAt=data.get("reportedAt"),
            remainingTime=data.get("remainingTime"),
            grid=[PhantomCell(**entry) for entry in data.get("grid", [])],
        )


@dataclass(slots=True)
class CellObservation:
    row: int
    col: int
    frame_index: int
    reveal_score: float
    sharpness: float
    crop_path: str | None = None
    state: str = "fully-revealed"
    quality_score: float = 0.0
    visible_width_ratio: float = 0.0
    occupancy_ratio: float = 0.0
    symbol_id: int | None = None
    image_url: str | None = None
    alternate_candidates: list[dict] = field(default_factory=list)
    discarded_candidates: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)

    def cell_id(self) -> str:
        return f"r{self.row:02d}c{self.col:02d}"


@dataclass(slots=True)
class MatchGroup:
    label: str
    members: list[str]
    confidence: float
    ambiguous: bool = False
    group_size: int = 2
    click_order: list[str] = field(default_factory=list)
    click_order_positions: list[dict] = field(default_factory=list)
    similarity_min: float = 0.0
    similarity_mean: float = 0.0


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
    candidate_debug_path: str | None = None
    solve_order_path: str | None = None
    board_source: dict | None = None

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
            "candidate_debug_path": self.candidate_debug_path,
            "solve_order_path": self.solve_order_path,
            "board_source": self.board_source,
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
                    click_order=group.get("click_order", group["members"]),
                    click_order_positions=group.get("click_order_positions", []),
                    similarity_min=group.get("similarity_min", 0.0),
                    similarity_mean=group.get("similarity_mean", 0.0),
                )
                for group in raw_groups
            ],
            unresolved=data["unresolved"],
            session_dir=data["session_dir"],
            grid_composed_path=data.get("grid_composed_path"),
            grid_debug_path=data.get("grid_debug_path"),
            grid_fit_debug_path=data.get("grid_fit_debug_path"),
            candidate_debug_path=data.get("candidate_debug_path"),
            solve_order_path=data.get("solve_order_path"),
            board_source=data.get("board_source"),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ReconstructionResult":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
