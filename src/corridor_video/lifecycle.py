from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid

from .models import VideoEvent


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    event_type: str
    reason_code: str
    detection_label: str
    entity_id: str | None
    bbox: tuple[float, float, float, float] | None
    first_seen: datetime
    last_seen: datetime
    hits: int = 1
    misses: int = 0
    active: bool = False


class EventLifecycle:
    """Turn frame-level candidates into one alarm per abnormal episode."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.confirm_frames = max(1, int(settings.get("confirm_frames", 3)))
        self.clear_frames = max(1, int(settings.get("clear_frames", 15)))
        self.clear_seconds = max(0.0, float(settings.get("clear_seconds", 0.0)))
        self.iou_threshold = float(settings.get("spatial_iou_threshold", 0.35))
        self.immediate_types = set(settings.get("immediate_types", ["abandoned_object", "worker_detected"]))
        self._incidents: dict[str, Incident] = {}

    def observe(self, camera_id: str, candidates: list[VideoEvent], at: datetime) -> list[VideoEvent]:
        """Return only candidates whose incident has just become active."""
        camera_incidents = [item for item in self._incidents.values() if item.camera_id == camera_id]
        unmatched = {item.incident_id for item in camera_incidents}
        emitted: list[VideoEvent] = []

        for event in candidates:
            bbox = self._event_bbox(event)
            entity_id = self._entity_id(event)
            detection_label = self._detection_label(event)
            incident = self._best_match(event, detection_label, entity_id, bbox, camera_incidents, unmatched)
            if incident is None:
                incident = Incident(
                    incident_id=f"INC-{uuid.uuid4().hex[:12].upper()}",
                    camera_id=camera_id,
                    event_type=event.event_type,
                    reason_code=event.reason_code,
                    detection_label=detection_label,
                    entity_id=entity_id,
                    bbox=bbox,
                    first_seen=at,
                    last_seen=at,
                )
                self._incidents[incident.incident_id] = incident
                camera_incidents.append(incident)
            else:
                unmatched.discard(incident.incident_id)
                incident.hits += 1
                incident.misses = 0
                incident.last_seen = at
                incident.bbox = bbox or incident.bbox
                incident.entity_id = entity_id or incident.entity_id

            required = 1 if event.event_type in self.immediate_types else self.confirm_frames
            if not incident.active and incident.hits >= required:
                incident.active = True
                event.metadata.update({
                    "incident_id": incident.incident_id,
                    "lifecycle_state": "active",
                    "first_seen": incident.first_seen.isoformat(timespec="milliseconds"),
                    "confirmed_frames": incident.hits,
                })
                emitted.append(event)

        for incident_id in unmatched:
            incident = self._incidents.get(incident_id)
            if incident is None:
                continue
            incident.misses += 1
            missing_seconds = (at - incident.last_seen).total_seconds()
            if incident.misses >= self.clear_frames and missing_seconds >= self.clear_seconds:
                self._incidents.pop(incident_id, None)
        return emitted

    def snapshot(self, camera_id: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "incident_id": item.incident_id,
                "camera_id": item.camera_id,
                "event_type": item.event_type,
                "reason_code": item.reason_code,
                "detection_label": item.detection_label,
                "entity_id": item.entity_id,
                "bbox_xyxy": list(item.bbox) if item.bbox is not None else None,
                "state": "active" if item.active else "candidate",
                "hits": item.hits,
                "misses": item.misses,
                "first_seen": item.first_seen.isoformat(timespec="milliseconds"),
                "last_seen": item.last_seen.isoformat(timespec="milliseconds"),
            }
            for item in self._incidents.values()
            if camera_id is None or item.camera_id == camera_id
        ]

    def defer(self, incident_id: str) -> None:
        """Re-arm an incident when an external cooldown postpones delivery."""
        incident = self._incidents.get(incident_id)
        if incident is not None:
            incident.active = False

    def _best_match(
        self,
        event: VideoEvent,
        detection_label: str,
        entity_id: str | None,
        bbox: tuple[float, float, float, float] | None,
        incidents: list[Incident],
        available: set[str],
    ) -> Incident | None:
        compatible = [
            item for item in incidents
            if item.incident_id in available
            and item.event_type == event.event_type
            and item.reason_code == event.reason_code
            and item.detection_label == detection_label
        ]
        if entity_id:
            same_entity = next((item for item in compatible if item.entity_id == entity_id), None)
            if same_entity is not None:
                return same_entity
        if bbox is None:
            return next((item for item in compatible if item.bbox is None), None)
        scored = [
            (max(self._iou(bbox, item.bbox), self._overlap_smaller(bbox, item.bbox) * 0.75), item)
            for item in compatible if item.bbox is not None
        ]
        score, best = max(scored, default=(0.0, None), key=lambda pair: pair[0])
        return best if score >= self.iou_threshold else None

    @staticmethod
    def _entity_id(event: VideoEvent) -> str | None:
        value = event.metadata.get("person_track_id") or event.metadata.get("object_track_id")
        return str(value) if value else None

    @staticmethod
    def _detection_label(event: VideoEvent) -> str:
        if not event.detections:
            return event.event_type
        return str(event.detections[0].get("label", event.event_type))

    @staticmethod
    def _event_bbox(event: VideoEvent) -> tuple[float, float, float, float] | None:
        if not event.detections:
            return None
        raw = event.detections[0].get("bbox_xyxy")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        return tuple(float(value) for value in raw)

    @staticmethod
    def _iou(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float] | None,
    ) -> float:
        if right is None:
            return 0.0
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _overlap_smaller(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float] | None,
    ) -> float:
        if right is None:
            return 0.0
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        smaller = min(left_area, right_area)
        return intersection / smaller if smaller > 0 else 0.0
