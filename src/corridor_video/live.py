from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import Any
from urllib.parse import urlparse
import json

import cv2

from .cli import build_standalone_engine
from .config import Settings
from .models import CameraObservation
from .runner import safe_source_name
from .vision import NativeVisionPipeline


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web"
ALARM_EVENT_TYPES = {
    "person_intrusion", "fire_smoke", "waterlogging", "abandoned_object",
    "other_intrusion", "abnormal_behavior", "visual_review_required",
}


class LiveRuntime:
    """Continuous file/RTSP detector backing the browser monitoring page."""

    def __init__(self, settings: Settings, source: str, camera_id: str, corridor_id: str,
                 loop_file: bool = True) -> None:
        self.settings = settings
        self.source, self.camera_id, self.corridor_id = source, camera_id, corridor_id
        self.loop_file = loop_file
        self.pipeline = NativeVisionPipeline({"baseline": settings.section("baseline"), **settings.section("vision")})
        self.engine = build_standalone_engine(settings, camera_id, corridor_id)
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._stop = Event()
        self._jpeg: bytes | None = None
        self._frame_sequence = 0
        self._events: list[dict[str, Any]] = []
        self._state: dict[str, Any] = {
            "status": "starting", "source": safe_source_name(source), "camera_id": camera_id,
            "corridor_id": corridor_id, "frames": 0, "fps": 0.0, "stable_change": False,
            "changed_ratio": 0.0, "baseline_state": "uninitialized", "baseline_version": 0,
            "detections": {"persons": 0, "objects": 0, "fire": 0, "water": 0},
            "active_incidents": [], "events": [], "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._thread = Thread(target=self._run, name=f"live-{camera_id}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=5)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, ensure_ascii=False))

    def wait_frame(self, after_sequence: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_sequence > after_sequence or self._stop.is_set(), timeout=timeout
            )
            return self._frame_sequence, self._jpeg

    def _run(self) -> None:
        capture = None
        file_source = Path(self.source).is_file()
        processed, started, next_frame_at = 0, monotonic(), monotonic()
        while not self._stop.is_set():
            if capture is None:
                capture = cv2.VideoCapture(self.source)
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    self._set_status("reconnecting", "无法打开视频源，正在重试")
                    self._stop.wait(2.0)
                    continue
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                if not source_fps or source_fps <= 0 or source_fps > 120:
                    source_fps = 25.0
                frame_interval, next_frame_at = 1.0 / source_fps, monotonic()
                self._set_status("running", None)

            ok, frame = capture.read()
            if not ok:
                if file_source and self.loop_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # A repeated test file is a new playback session. Clear
                    # tracks/incidents so stale boxes from the previous loop
                    # cannot remain over the new baseline.
                    self.pipeline = NativeVisionPipeline({
                        "baseline": self.settings.section("baseline"), **self.settings.section("vision")
                    })
                    self.engine = build_standalone_engine(
                        self.settings, self.camera_id, self.corridor_id
                    )
                    next_frame_at = monotonic()
                    continue
                capture.release()
                capture = None
                self.pipeline.invalidate(self.camera_id)
                now = datetime.now(timezone.utc)
                self.engine.process(CameraObservation(
                    self.camera_id, self.corridor_id, now, 0, 0, False, healthy=False
                ))
                if file_source:
                    self._set_status("ended", None)
                    break
                self._set_status("reconnecting", "视频流中断，正在重新连接")
                self._stop.wait(2.0)
                continue

            now = datetime.now(timezone.utc)
            session_active = self.corridor_id in self.engine.sessions
            self.pipeline.freeze(self.camera_id) if session_active else self.pipeline.unfreeze(self.camera_id)
            observation = self.pipeline.analyze(
                frame, self.camera_id, self.corridor_id, now,
                allow_baseline_update=not session_active,
            )
            new_events = self.engine.process(observation)
            processed += 1
            elapsed = max(monotonic() - started, 0.001)
            incidents = self.engine.lifecycle.snapshot(self.camera_id)
            engine_state = self.engine.snapshot(self.corridor_id)
            session_state = engine_state.get("session") or {}
            tracked_objects = [item for item in session_state.get("objects", [])
                               if item.get("camera_id") == self.camera_id]
            annotated = self._annotate(frame, observation, incidents, tracked_objects)
            encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not encoded_ok:
                continue
            with self._condition:
                self._events.extend(event.to_dict() for event in new_events)
                self._events = self._events[-50:]
                self._state.update({
                    "status": "running", "frames": processed, "fps": round(processed / elapsed, 1),
                    "stable_change": observation.stable_change,
                    "changed_ratio": round(observation.changed_ratio, 5),
                    "baseline_state": observation.baseline_state.value,
                    "baseline_version": observation.baseline_version,
                    "detections": {
                        "persons": len(observation.persons), "objects": len(observation.objects),
                        "fire": len(observation.fire_candidates),
                        "water": int(observation.water_candidate is not None),
                    },
                    "active_incidents": [
                        item for item in incidents
                        if item["state"] == "active" and item["event_type"] in ALARM_EVENT_TYPES
                    ],
                    "events": list(reversed(self._events)), "error": None,
                })
                self._jpeg = encoded.tobytes()
                self._frame_sequence += 1
                self._condition.notify_all()

            if file_source:
                next_frame_at += frame_interval
                delay = next_frame_at - monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                elif delay < -1.0:
                    next_frame_at = monotonic()
        if capture is not None:
            capture.release()

    def _set_status(self, status: str, message: str | None) -> None:
        with self._condition:
            self._state["status"], self._state["error"] = status, message
            self._condition.notify_all()

    @staticmethod
    def _annotate(frame, observation, incidents: list[dict[str, Any]],
                  tracked_objects: list[dict[str, Any]] | None = None):
        output = frame.copy()
        for person in observation.persons:
            if not LiveRuntime._covered_by_alarm(person.bbox, "person", person.track_id, incidents):
                LiveRuntime._box(output, person.bbox, f"PERSON {person.helmet.value} {person.confidence:.2f}", (40, 210, 255), 2)
        for item in observation.objects:
            if not LiveRuntime._covered_by_alarm(item.bbox, item.category, item.track_id, incidents):
                LiveRuntime._box(output, item.bbox, f"{item.category} {item.confidence:.2f}", (0, 190, 255), 2)
        # Keep the last confirmed position visible during a short detector
        # dropout. This is a tracked candidate, not an alarm (alarms stay red).
        observed_boxes = [item.bbox for item in observation.objects]
        for item in tracked_objects or []:
            raw = item.get("bbox_xyxy")
            if not raw or any(LiveRuntime._iou(raw, (
                    box.x1, box.y1, box.x2, box.y2)) >= 0.35 for box in observed_boxes):
                continue
            LiveRuntime._box(output, raw, "TRACKED OBJECT", (0, 190, 255), 2)
        for item in observation.fire_candidates:
            if item.bbox:
                LiveRuntime._box(output, item.bbox, f"{item.label.upper()} {item.confidence:.2f}", (0, 80, 255), 2)
        if observation.water_candidate and observation.water_candidate.bbox:
            item = observation.water_candidate
            LiveRuntime._box(output, item.bbox, f"WATER {item.confidence:.2f}", (255, 170, 30), 2)
        for incident in incidents:
            raw = incident.get("bbox_xyxy")
            if (raw and incident.get("misses", 0) == 0
                    and incident.get("event_type") in ALARM_EVENT_TYPES):
                LiveRuntime._box(output, raw, f"ALARM {incident['detection_label'].upper()}", (0, 0, 255), 4)
        status = f"BASE v{observation.baseline_version} {observation.baseline_state.value}  DIFF {observation.changed_ratio * 100:.1f}%"
        cv2.rectangle(output, (0, 0), (min(output.shape[1], 650), 42), (7, 16, 24), -1)
        cv2.putText(output, status, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 226, 208), 2)
        return output

    @staticmethod
    def _box(frame, box, label: str, color: tuple[int, int, int], thickness: int) -> None:
        coords = (box.x1, box.y1, box.x2, box.y2) if hasattr(box, "x1") else box
        x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        top = max(0, y1 - text_size[1] - 12)
        cv2.rectangle(frame, (x1, top), (min(frame.shape[1], x1 + text_size[0] + 10), y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, max(text_size[1] + 2, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    @staticmethod
    def _covered_by_alarm(box, label: str, entity_id: str | None,
                          incidents: list[dict[str, Any]]) -> bool:
        left = (box.x1, box.y1, box.x2, box.y2) if hasattr(box, "x1") else tuple(box)
        for incident in incidents:
            if incident.get("state") != "active":
                continue
            if incident.get("event_type") not in ALARM_EVENT_TYPES:
                continue
            if incident.get("misses", 0) != 0:
                continue
            incident_label = str(incident.get("detection_label", ""))
            labels_match = incident_label == label or {incident_label, label} <= {"object", "portable_object"}
            if not labels_match:
                continue
            if entity_id and incident.get("entity_id") == entity_id:
                return True
            right = incident.get("bbox_xyxy")
            if right:
                iou = LiveRuntime._iou(left, right)
                overlap = LiveRuntime._overlap_smaller(left, right)
                if iou >= 0.20 or overlap >= 0.20:
                    return True
        return False

    @staticmethod
    def _iou(left, right) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _overlap_smaller(left, right) -> float:
        x1, y1 = max(left[0], right[0]), max(left[1], right[1])
        x2, y2 = min(left[2], right[2]), min(left[3], right[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
        right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
        smaller = min(left_area, right_area)
        return intersection / smaller if smaller > 0 else 0.0


def serve_live(settings: Settings, source: str, camera_id: str, corridor_id: str,
               host: str = "127.0.0.1", port: int = 8766, loop_file: bool = True) -> None:
    runtime = LiveRuntime(settings, source, camera_id, corridor_id, loop_file)

    class LiveHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/live/state":
                return self._json(runtime.state())
            if path == "/stream.mjpg":
                return self._stream()
            file = STATIC / ("live.html" if path == "/" else path.lstrip("/"))
            if file.is_file() and STATIC in file.resolve().parents:
                content_type = "text/css" if file.suffix == ".css" else "text/javascript" if file.suffix == ".js" else "text/html"
                payload = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while not runtime._stop.is_set():
                    sequence, payload = runtime.wait_frame(sequence)
                    if payload is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(payload)).encode("ascii") + b"\r\n\r\n")
                    self.wfile.write(payload + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def _json(self, value: dict[str, Any]) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), LiveHandler)
    print(f"持续检测页面: http://{host}:{port}  来源: {safe_source_name(source)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        runtime.close()
