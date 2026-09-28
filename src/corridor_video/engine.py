from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from typing import Any
import uuid

from .evidence import EvidenceStore
from .lifecycle import EventLifecycle
from .models import BaselineState, CameraObservation, HelmetState, ObjectObservation, VideoEvent
from .outbox import EventOutbox
from .topology import Topology
from .vlm import OpenAICompatibleVisionReviewer


@dataclass
class CameraState:
    healthy: bool = False
    baseline_state: BaselineState = BaselineState.UNINITIALIZED
    baseline_version: int = 0
    last_seen: datetime | None = None
    last_person_at: datetime | None = None


@dataclass
class TrackedObject:
    observation: ObjectObservation
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    alone_since: datetime | None = None


@dataclass
class CorridorSession:
    session_id: str
    corridor_id: str
    started_at: datetime
    last_person_at: datetime
    frozen_baselines: dict[str, int]
    state: str = "active"
    objects: dict[str, TrackedObject] = field(default_factory=dict)


class EventEngine:
    """The only component allowed to convert visual facts into events."""

    def __init__(self, settings: dict[str, Any], topology: Topology, outbox: EventOutbox,
                 evidence: EvidenceStore, reviewer: OpenAICompatibleVisionReviewer) -> None:
        self.settings, self.topology, self.outbox = settings, topology, outbox
        self.evidence, self.reviewer = evidence, reviewer
        self.cameras = {camera_id: CameraState() for camera_id in topology.cameras}
        self.sessions: dict[str, CorridorSession] = {}
        self.lifecycle = EventLifecycle(settings.get("lifecycle", {}))
        self._last_event: dict[str, datetime] = {}

    def process(self, observation: CameraObservation) -> list[VideoEvent]:
        camera = self.cameras.setdefault(observation.camera_id, CameraState())
        camera.healthy, camera.last_seen = observation.healthy, observation.captured_at
        camera.baseline_state, camera.baseline_version = observation.baseline_state, observation.baseline_version
        if not observation.healthy or observation.global_change:
            camera.baseline_state = BaselineState.INVALID
            session = self.sessions.get(observation.corridor_id)
            if session:
                session.state = "manual_review"
            return []

        events: list[VideoEvent] = []
        session = self.sessions.get(observation.corridor_id)
        if observation.persons:
            camera.last_person_at = observation.captured_at
            session = self._touch_session(observation)
            for person in observation.persons:
                if person.helmet == HelmetState.NOT_WEARING:
                    events.append(VideoEvent(
                        "person_intrusion", "检测到未正确佩戴安全帽人员", observation.captured_at,
                        observation.corridor_id, observation.camera_id, person.helmet_confidence,
                        "helmet_not_wearing", session.session_id,
                        [self._detection(person.bbox, "person", person.confidence, observation)],
                        metadata={"person_track_id": person.track_id, "appearance_based_worker_rule": True},
                    ))
                elif person.helmet == HelmetState.WEARING and self.settings["person"].get("emit_worker_detected", False):
                    events.append(VideoEvent(
                        "worker_detected", "检测到正确佩戴安全帽人员", observation.captured_at,
                        observation.corridor_id, observation.camera_id, person.helmet_confidence,
                        "helmet_wearing", session.session_id,
                        [self._detection(person.bbox, "person", person.confidence, observation)],
                        metadata={"person_track_id": person.track_id, "appearance_based_worker_rule": True},
                    ))

        if session:
            self._update_objects(session, observation)
            events.extend(self._advance_session(session, observation))
        events.extend(self._visual_events(observation))

        # Frame detections are reconciled into long-lived incidents here. An
        # active incident is emitted exactly once; following frames only keep
        # it alive until ``clear_frames`` consecutive misses close it.
        events = self.lifecycle.observe(observation.camera_id, events, observation.captured_at)

        accepted: list[VideoEvent] = []
        for event in events:
            key = self._dedup_key(event)
            if not self._cooldown_allows(key, event.captured_at):
                incident_id = event.metadata.get("incident_id")
                if incident_id:
                    self.lifecycle.defer(str(incident_id))
                continue
            event.images = self.evidence.preserve(observation.image_path, event.event_id, event.captured_at, event.detections)
            if self.outbox.enqueue(event, key):
                self._last_event[key] = event.captured_at
                accepted.append(event)
        return accepted

    def snapshot(self, corridor_id: str) -> dict[str, Any]:
        session = self.sessions.get(corridor_id)
        cameras = self.topology.corridors.get(corridor_id, [])
        return {
            "corridor_id": corridor_id,
            "session": None if not session else {
                "session_id": session.session_id,
                "state": session.state,
                "started_at": session.started_at.isoformat(),
                "last_person_at": session.last_person_at.isoformat(),
                "object_candidates": len(session.objects),
                "objects": [{
                    "track_id": key,
                    "camera_id": item.camera_id,
                    "category": item.observation.category,
                    "confidence": item.observation.confidence,
                    "bbox_xyxy": [item.observation.bbox.x1, item.observation.bbox.y1,
                                  item.observation.bbox.x2, item.observation.bbox.y2],
                    "first_seen": item.first_seen.isoformat(),
                    "last_seen": item.last_seen.isoformat(),
                    "alone_since": None if item.alone_since is None else item.alone_since.isoformat(),
                } for key, item in session.objects.items()],
                "frozen_baselines": session.frozen_baselines,
            },
            "cameras": {camera_id: {
                "healthy": self.cameras[camera_id].healthy,
                "baseline_state": self.cameras[camera_id].baseline_state.value,
                "baseline_version": self.cameras[camera_id].baseline_version,
                "incidents": self.lifecycle.snapshot(camera_id),
            } for camera_id in cameras},
        }

    def _touch_session(self, observation: CameraObservation) -> CorridorSession:
        session = self.sessions.get(observation.corridor_id)
        if session is None:
            camera_ids = self.topology.corridors.get(observation.corridor_id, [observation.camera_id])
            frozen = {camera_id: self.cameras[camera_id].baseline_version for camera_id in camera_ids}
            session = CorridorSession(f"CS-{uuid.uuid4().hex[:12].upper()}", observation.corridor_id,
                                      observation.captured_at, observation.captured_at, frozen)
            self.sessions[observation.corridor_id] = session
        session.last_person_at, session.state = observation.captured_at, "active"
        for camera_id in self.topology.corridors.get(observation.corridor_id, [observation.camera_id]):
            self.cameras[camera_id].baseline_state = BaselineState.FROZEN
        return session

    def _update_objects(self, session: CorridorSession, observation: CameraObservation) -> None:
        present: set[str] = set()
        for item in observation.objects:
            if not item.is_portable:
                continue
            key = f"{observation.camera_id}:{item.track_id}"
            tracked = session.objects.get(key)
            if tracked is None:
                matched_key = self._match_object(session, observation.camera_id, item, present)
                if matched_key is not None:
                    key = matched_key
                    tracked = session.objects[key]
            present.add(key)
            if tracked is None:
                tracked = TrackedObject(item, observation.camera_id, observation.captured_at, observation.captured_at)
                session.objects[key] = tracked
            else:
                tracked.observation, tracked.last_seen = item, observation.captured_at
            if session.state in {"exit_pending", "inspecting"} and not observation.persons:
                tracked.alone_since = tracked.alone_since or observation.captured_at
        # Difference masks can briefly lose a stationary box because of a
        # person crossing it, compression noise or illumination.  A single
        # missed frame must not destroy all accumulated persistence.  A truly
        # carried-away item is still removed after the bounded grace period.
        missing_grace = float(self.settings["session"].get("object_missing_grace_seconds", 2.0))
        stale = [key for key, item in session.objects.items()
                 if item.camera_id == observation.camera_id and key not in present
                 and not observation.persons
                 and (observation.captured_at - item.last_seen).total_seconds() >= missing_grace]
        for key in stale:
            session.objects.pop(key, None)

    def _match_object(self, session: CorridorSession, camera_id: str,
                      observation: ObjectObservation, claimed: set[str]) -> str | None:
        """Associate a detection spatially when the lightweight tracker changes ID."""
        minimum_iou = float(self.settings["session"].get("object_match_iou", 0.15))
        max_center = float(self.settings["session"].get("object_match_center_pixels", 120.0))
        best: tuple[float, str] | None = None
        for key, tracked in session.objects.items():
            if key in claimed or tracked.camera_id != camera_id:
                continue
            portable_labels = {"object", "portable_object", "box", "toolbox"}
            same_category = tracked.observation.category == observation.category
            if not same_category and not ({tracked.observation.category,
                                           observation.category} <= portable_labels):
                continue
            iou = self._iou(tracked.observation.bbox, observation.bbox)
            left, right = tracked.observation.bbox.center, observation.bbox.center
            distance = hypot(left[0] - right[0], left[1] - right[1])
            adaptive_distance = max(max_center, tracked.observation.bbox.width,
                                    tracked.observation.bbox.height)
            if iou < minimum_iou and distance > adaptive_distance:
                continue
            score = iou - distance / max(adaptive_distance, 1.0) * 0.1
            if best is None or score > best[0]:
                best = (score, key)
        return None if best is None else best[1]

    @staticmethod
    def _iou(left, right) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = left.width * left.height + right.width * right.height - intersection
        return intersection / union if union > 0 else 0.0

    def _advance_session(self, session: CorridorSession, observation: CameraObservation) -> list[VideoEvent]:
        camera_ids = self.topology.corridors.get(session.corridor_id, [observation.camera_id])
        stale_seconds = float(self.settings["session"].get("camera_stale_seconds", 10.0))
        unavailable = [camera_id for camera_id in camera_ids if self._camera_unavailable(camera_id, observation.captured_at, stale_seconds)]
        if unavailable and self.settings["session"].get("offline_blocks_abandoned", True):
            session.state = "manual_review"
            return []
        latest_person = max((self.cameras[camera_id].last_person_at for camera_id in camera_ids
                             if self.cameras[camera_id].last_person_at), default=session.last_person_at)
        session.last_person_at = latest_person
        exit_seconds = float(self.settings["session"]["exit_confirmation_seconds"])
        if (observation.captured_at - latest_person).total_seconds() < exit_seconds:
            session.state = "exit_pending" if not observation.persons else "active"
            return []
        session.state = "inspecting"
        persistence = float(self.settings["session"]["abandoned_persistence_seconds"])
        events: list[VideoEvent] = []
        for key, tracked in list(session.objects.items()):
            if tracked.camera_id != observation.camera_id or tracked.last_seen != observation.captured_at:
                continue
            if tracked.alone_since is None or (observation.captured_at - tracked.alone_since).total_seconds() < persistence:
                continue
            events.append(VideoEvent(
                "abandoned_object", f"人员离开廊段后仍存在新增物品：{tracked.observation.category}",
                observation.captured_at, session.corridor_id, tracked.camera_id, tracked.observation.confidence,
                "persisted_after_corridor_exit", session.session_id,
                [self._detection(tracked.observation.bbox, tracked.observation.category,
                                 tracked.observation.confidence, observation)],
                metadata={"object_track_id": key, "frozen_baselines": session.frozen_baselines,
                          "persistence_seconds": persistence},
            ))
            session.objects.pop(key, None)
        if events or (not session.objects and session.state == "inspecting"):
            session.state = "closed"
            self.sessions.pop(session.corridor_id, None)
            for camera_id in camera_ids:
                self.cameras[camera_id].baseline_state = BaselineState.VALID
        return events

    def _visual_events(self, observation: CameraObservation) -> list[VideoEvent]:
        events: list[VideoEvent] = []
        if observation.fire_candidates:
            hints = [self._detection(item.bbox, item.label, item.confidence, observation) for item in observation.fire_candidates]
            review = self.reviewer.review("fire_smoke_review", [observation.image_path] if observation.image_path else [], hints)
            best = max(observation.fire_candidates, key=lambda item: item.confidence)
            if review.label in {"fire", "smoke"}:
                events.append(VideoEvent("fire_smoke", "视觉复核确认火焰或烟雾", observation.captured_at,
                    observation.corridor_id, observation.camera_id, review.confidence, "vlm_confirmed", detections=hints,
                    metadata={"review_status": review.status, "review_label": review.label}))
            elif review.status != "succeeded":
                events.append(VideoEvent("visual_review_required", "发现火烟候选，等待视觉复核", observation.captured_at,
                    observation.corridor_id, observation.camera_id, best.confidence, "review_unavailable", detections=hints,
                    metadata={"review_status": review.status, "candidate_label": best.label}))
        if observation.water_candidate:
            item = observation.water_candidate
            hints = [self._detection(item.bbox, item.label, item.confidence, observation)]
            review = self.reviewer.review("waterlogging_review", [observation.image_path] if observation.image_path else [], hints)
            label = "waterlogging" if review.label == "waterlogging" else "visual_review_required"
            events.append(VideoEvent(label, "检测到积水" if label == "waterlogging" else "发现积水候选，等待视觉复核",
                observation.captured_at, observation.corridor_id, observation.camera_id,
                review.confidence if label == "waterlogging" else item.confidence,
                "vlm_confirmed" if label == "waterlogging" else "review_unavailable", detections=hints,
                metadata={"review_status": review.status, "no_depth_inference": True}))
        return events

    def _camera_unavailable(self, camera_id: str, now: datetime, stale_seconds: float) -> bool:
        state = self.cameras.get(camera_id)
        return state is None or not state.healthy or state.last_seen is None or (now - state.last_seen).total_seconds() > stale_seconds

    @staticmethod
    def _detection(box, label: str, confidence: float, observation: CameraObservation) -> dict:
        coords = None if box is None else [box.x1, box.y1, box.x2, box.y2]
        return {"label": label, "confidence": confidence, "bbox_xyxy": coords,
                "source_width": observation.frame_width, "source_height": observation.frame_height}

    @staticmethod
    def _dedup_key(event: VideoEvent) -> str:
        if event.event_type == "abandoned_object":
            return f"{event.session_id}:{event.metadata.get('object_track_id')}"
        return f"{event.camera_id}:{event.event_type}:{event.reason_code}"

    def _cooldown_allows(self, key: str, now: datetime) -> bool:
        previous = self._last_event.get(key) or self.outbox.last_captured_at(key)
        return previous is None or (now - previous).total_seconds() >= float(self.settings["events"]["cooldown_seconds"])
