from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
import uuid


class HelmetState(StrEnum):
    WEARING = "wearing"
    NOT_WEARING = "not_wearing"
    UNKNOWN = "unknown"


class BaselineState(StrEnum):
    UNINITIALIZED = "uninitialized"
    CALIBRATING = "calibrating"
    VALID = "valid"
    FROZEN = "frozen"
    INVALID = "invalid"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class PersonObservation:
    track_id: str
    bbox: BoundingBox
    confidence: float
    helmet: HelmetState = HelmetState.UNKNOWN
    helmet_confidence: float = 0.0


@dataclass(frozen=True)
class ObjectObservation:
    track_id: str
    category: str
    bbox: BoundingBox
    confidence: float
    is_portable: bool = False


@dataclass(frozen=True)
class VisualCandidate:
    label: str
    bbox: BoundingBox | None
    confidence: float


@dataclass(frozen=True)
class CameraObservation:
    camera_id: str
    corridor_id: str
    captured_at: datetime
    frame_width: int
    frame_height: int
    stable_change: bool
    healthy: bool = True
    persons: tuple[PersonObservation, ...] = ()
    objects: tuple[ObjectObservation, ...] = ()
    fire_candidates: tuple[VisualCandidate, ...] = ()
    water_candidate: VisualCandidate | None = None
    image_path: str | None = None
    baseline_version: int = 0
    baseline_state: BaselineState = BaselineState.UNINITIALIZED
    changed_ratio: float = 0.0
    global_change: bool = False


@dataclass
class VideoEvent:
    event_type: str
    event_text: str
    captured_at: datetime
    corridor_id: str
    camera_id: str
    confidence: float
    reason_code: str
    session_id: str | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"VID-{uuid.uuid4().hex.upper()}")
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["captured_at"] = self.captured_at.isoformat(timespec="milliseconds")
        return value
