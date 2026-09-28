from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from corridor_video.baseline import DynamicBaseline
from corridor_video.config import load_settings
from corridor_video.demo import DemoRuntime
from corridor_video.delivery import to_weknora_payload
from corridor_video.engine import EventEngine
from corridor_video.evidence import EvidenceStore
from corridor_video.lifecycle import EventLifecycle
from corridor_video.models import BaselineState, BoundingBox, CameraObservation, HelmetState, ObjectObservation, PersonObservation, VideoEvent
from corridor_video.outbox import EventOutbox
from corridor_video.runner import safe_source_name
from corridor_video.topology import load_topology
from corridor_video.vision import NativeVisionPipeline
from corridor_video.vlm import OpenAICompatibleVisionReviewer

ROOT = Path(__file__).resolve().parents[1]


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(ROOT / "config" / "default.toml")

    def test_dynamic_baseline_requires_stable_change(self):
        gate = DynamicBaseline(stable_confirm_frames=2, clear_confirm_frames=1, global_change_ratio=.9, learning_rate=0)
        gate.process(np.zeros((20, 20), np.uint8))
        changed = np.zeros((20, 20), np.uint8)
        changed[:8] = 255
        self.assertFalse(gate.process(changed).changed)
        self.assertTrue(gate.process(changed).changed)

    def test_difference_is_against_baseline_not_previous_frame(self):
        gate = DynamicBaseline(stable_confirm_frames=1, clear_confirm_frames=1,
                               global_change_ratio=.9, learning_rate=0)
        baseline = np.zeros((20, 20), np.uint8)
        changed = baseline.copy()
        changed[:8] = 255
        gate.process(baseline)
        self.assertTrue(gate.process(changed).changed)
        # A previous-frame differencer would flag this transition. Baseline
        # comparison correctly sees that the scene has returned to reference.
        self.assertFalse(gate.process(baseline).changed)

    def test_candidate_change_is_not_learned_into_baseline(self):
        gate = DynamicBaseline(stable_confirm_frames=3, clear_confirm_frames=1,
                               global_change_ratio=.9, learning_rate=.5)
        baseline = np.zeros((20, 20), np.uint8)
        changed = baseline.copy()
        changed[:8] = 80
        gate.process(baseline)
        self.assertFalse(gate.process(changed).changed)
        self.assertFalse(gate.process(changed).changed)
        self.assertTrue(gate.process(changed).changed)

    def test_rtsp_password_is_not_exposed_in_results(self):
        safe = safe_source_name("rtsp://operator:secret@10.0.0.8:554/Streaming/Channels/101?token=x")
        self.assertEqual(safe, "rtsp://10.0.0.8:554/Streaming/Channels/101")

    def test_lifecycle_emits_once_while_incident_continues(self):
        lifecycle = EventLifecycle({"confirm_frames": 2, "clear_frames": 3, "spatial_iou_threshold": .3})
        start = datetime.now(timezone.utc)
        emitted = []
        for index in range(50):
            emitted.extend(lifecycle.observe("CAM-A", [self._event(start + timedelta(seconds=index))],
                                                     start + timedelta(seconds=index)))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].metadata["confirmed_frames"], 2)
        self.assertEqual(lifecycle.snapshot("CAM-A")[0]["state"], "active")

    def test_lifecycle_reopens_only_after_confirmed_clear(self):
        lifecycle = EventLifecycle({"confirm_frames": 2, "clear_frames": 3, "spatial_iou_threshold": .3})
        start = datetime.now(timezone.utc)
        first = []
        for index in range(2):
            first.extend(lifecycle.observe("CAM-A", [self._event(start + timedelta(seconds=index))],
                                                   start + timedelta(seconds=index)))
        for index in range(2, 5):
            lifecycle.observe("CAM-A", [], start + timedelta(seconds=index))
        second = []
        for index in range(5, 7):
            second.extend(lifecycle.observe("CAM-A", [self._event(start + timedelta(seconds=index))],
                                                     start + timedelta(seconds=index)))
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].metadata["incident_id"], second[0].metadata["incident_id"])

    @staticmethod
    def _event(at):
        return VideoEvent(
            "fire_smoke", "测试火焰", at, "CORRIDOR-01", "CAM-A", .9, "vlm_confirmed",
            detections=[{"label": "fire", "confidence": .9, "bbox_xyxy": [10, 10, 60, 80]}],
        )

    def test_native_vision_detects_worker_without_weights(self):
        pipeline = NativeVisionPipeline({"baseline": self.settings.section("baseline"), **self.settings.section("vision")})
        blank = DemoRuntime._draw_frame("CAM-A", [])
        worker = DemoRuntime._draw_frame("CAM-A", ["worker"])
        now = datetime.now(timezone.utc)
        for index in range(5):
            pipeline.analyze(blank, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=index))
        result = None
        for index in range(3):
            result = pipeline.analyze(worker, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=6 + index))
        self.assertTrue(result.stable_change)
        self.assertEqual(len(result.persons), 1)
        self.assertEqual(result.persons[0].helmet, HelmetState.WEARING)

    def test_geometry_only_person_candidate_is_not_emitted(self):
        pipeline = NativeVisionPipeline({"baseline": self.settings.section("baseline"), **self.settings.section("vision")})
        blank = DemoRuntime._draw_frame("CAM-A", [])
        person_without_visible_helmet = blank.copy()
        cv2.rectangle(person_without_visible_helmet, (180, 135), (230, 280), (190, 90, 45), -1)
        now = datetime.now(timezone.utc)
        for index in range(5):
            pipeline.analyze(blank, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=index))
        result = None
        for index in range(3):
            result = pipeline.analyze(person_without_visible_helmet, "CAM-A", "CORRIDOR-01",
                                      now + timedelta(seconds=6 + index))
        self.assertEqual(len(result.persons), 0)

    def test_semantic_detection_is_limited_to_changed_region(self):
        pipeline = NativeVisionPipeline({"baseline": self.settings.section("baseline"), **self.settings.section("vision")})
        baseline = DemoRuntime._draw_frame("CAM-A", [])
        cv2.rectangle(baseline, (300, 180), (630, 245), (180, 110, 30), -1)  # static pipe-like structure
        current = baseline.copy()
        cv2.rectangle(current, (180, 135), (230, 280), (190, 90, 45), -1)
        cv2.rectangle(current, (180, 125), (230, 154), (0, 0, 230), -1)
        now = datetime.now(timezone.utc)
        for index in range(5):
            pipeline.analyze(baseline, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=index))
        result = None
        for index in range(3):
            result = pipeline.analyze(current, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=6 + index))
        self.assertEqual(len(result.persons), 1)
        self.assertEqual(len(result.objects), 0)

    def test_top_timestamp_region_is_ignored(self):
        pipeline = NativeVisionPipeline({"baseline": self.settings.section("baseline"), **self.settings.section("vision")})
        baseline = DemoRuntime._draw_frame("CAM-A", [])
        timestamp_changed = baseline.copy()
        cv2.rectangle(timestamp_changed, (0, 0), (640, 30), (0, 220, 250), -1)
        now = datetime.now(timezone.utc)
        for index in range(5):
            pipeline.analyze(baseline, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=index))
        result = None
        for index in range(3):
            result = pipeline.analyze(timestamp_changed, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=6 + index))
        self.assertTrue(result.stable_change)
        self.assertEqual(len(result.persons) + len(result.objects) + len(result.fire_candidates), 0)

    def test_cardboard_box_is_split_from_connected_vertical_change(self):
        pipeline = NativeVisionPipeline({"baseline": self.settings.section("baseline"),
                                         **self.settings.section("vision")})
        baseline = np.full((360, 640, 3), 95, dtype=np.uint8)
        current = baseline.copy()
        # The thin line models a cable-rack/floor edge joined to the carton in
        # the raw difference mask. The box must still become its own object.
        cardboard = (100, 130, 180)
        cv2.rectangle(current, (428, 150), (430, 300), cardboard, -1)
        cv2.rectangle(current, (400, 240), (460, 310), cardboard, -1)
        now = datetime.now(timezone.utc)
        for index in range(5):
            pipeline.analyze(baseline, "CAM-A", "CORRIDOR-01", now + timedelta(seconds=index))
        result = None
        for index in range(3):
            result = pipeline.analyze(current, "CAM-A", "CORRIDOR-01",
                                      now + timedelta(seconds=6 + index))
        self.assertTrue(any(item.bbox.height < 100 for item in result.objects))

    def test_abandoned_requires_corridor_exit_and_persistence(self):
        settings = deepcopy(self.settings.values)
        settings["session"].update(exit_confirmation_seconds=2, abandoned_persistence_seconds=2, camera_stale_seconds=100)
        with TemporaryDirectory() as folder:
            engine = self._engine(settings, folder)
            start = datetime.now(timezone.utc)
            self._prime(engine, start)
            person = PersonObservation("P1", BoundingBox(10, 10, 40, 100), .9, HelmetState.WEARING, .9)
            item = ObjectObservation("O1", "toolbox", BoundingBox(80, 80, 120, 110), .8, True)
            engine.process(self._obs("CAM-A", start + timedelta(seconds=1), persons=(person,)))
            engine.process(self._obs("CAM-B", start + timedelta(seconds=1), objects=(item,)))
            self.assertEqual(engine.outbox.count(), 1)  # worker status only
            for second in (3, 5):
                engine.process(self._obs("CAM-A", start + timedelta(seconds=second)))
                events = engine.process(self._obs("CAM-B", start + timedelta(seconds=second), objects=(item,)))
                engine.process(self._obs("CAM-C", start + timedelta(seconds=second)))
            abandoned = [event for event in events if event.event_type == "abandoned_object"]
            self.assertEqual(len(abandoned), 1)
            self.assertEqual(abandoned[0].camera_id, "CAM-B")

    def test_carried_away_object_never_alerts(self):
        settings = deepcopy(self.settings.values)
        settings["session"].update(exit_confirmation_seconds=2, abandoned_persistence_seconds=2, camera_stale_seconds=100)
        with TemporaryDirectory() as folder:
            engine = self._engine(settings, folder)
            start = datetime.now(timezone.utc)
            self._prime(engine, start)
            person = PersonObservation("P1", BoundingBox(10, 10, 40, 100), .9, HelmetState.WEARING, .9)
            item = ObjectObservation("O1", "toolbox", BoundingBox(80, 80, 120, 110), .8, True)
            engine.process(self._obs("CAM-A", start + timedelta(seconds=1), persons=(person,)))
            engine.process(self._obs("CAM-B", start + timedelta(seconds=1), objects=(item,)))
            produced = []
            for second in (3, 5, 8):
                for camera_id in ("CAM-A", "CAM-B", "CAM-C"):
                    produced.extend(engine.process(self._obs(camera_id, start + timedelta(seconds=second))))
            self.assertFalse(any(event.event_type == "abandoned_object" for event in produced))

    def test_object_survives_short_dropout_and_track_id_change(self):
        settings = deepcopy(self.settings.values)
        settings["session"].update(
            exit_confirmation_seconds=2,
            abandoned_persistence_seconds=2,
            object_missing_grace_seconds=3,
            object_match_center_pixels=80,
            camera_stale_seconds=100,
        )
        with TemporaryDirectory() as folder:
            engine = self._engine(settings, folder)
            start = datetime.now(timezone.utc)
            self._prime(engine, start)
            person = PersonObservation("P1", BoundingBox(10, 10, 40, 100), .9,
                                       HelmetState.WEARING, .9)
            first = ObjectObservation("O1", "portable_object",
                                      BoundingBox(80, 80, 120, 120), .8, True)
            shifted = ObjectObservation("O99", "portable_object",
                                        BoundingBox(84, 82, 124, 122), .82, True)
            engine.process(self._obs("CAM-A", start + timedelta(seconds=1), persons=(person,)))
            engine.process(self._obs("CAM-B", start + timedelta(seconds=1), objects=(first,)))
            # One missing frame must not erase the candidate.
            engine.process(self._obs("CAM-B", start + timedelta(seconds=2)))
            self.assertEqual(len(engine.sessions["CORRIDOR-01"].objects), 1)
            # A nearby detection with a new tracker ID is the same object.
            engine.process(self._obs("CAM-B", start + timedelta(seconds=3), objects=(shifted,)))
            self.assertEqual(len(engine.sessions["CORRIDOR-01"].objects), 1)
            events = engine.process(self._obs("CAM-B", start + timedelta(seconds=5), objects=(shifted,)))
            self.assertEqual(len([event for event in events
                                  if event.event_type == "abandoned_object"]), 1)

    def test_full_dynamic_demo_emits_one_abandoned_event(self):
        demo = DemoRuntime()
        try:
            result = None
            for _ in range(35):
                result = demo.step()
            abandoned = [event for event in result["events"] if event["event_type"] == "abandoned_object"]
            self.assertEqual(len(abandoned), 1)
            self.assertEqual(abandoned[0]["camera_id"], "CAM-B")
            self.assertEqual(result["metrics"]["llm_calls"], 0)
        finally:
            demo.close()

    def test_weknora_contract_translation(self):
        payload = to_weknora_payload({
            "schema_version": "1.0", "event_id": "VID-1234567890ABCDEF", "event_text": "测试事件",
            "event_type": "abandoned_object", "captured_at": "2026-09-10T09:00:24.000+08:00",
            "corridor_id": "CORRIDOR-01", "camera_id": "CAM-B", "confidence": .8,
            "reason_code": "persisted_after_corridor_exit", "session_id": "CS-1",
            "detections": [], "images": [{"local_path": "must-not-leak.jpg"}], "metadata": {},
        })
        self.assertEqual(payload["event_time"], "2026-09-10T09:00:24.000+08:00")
        self.assertEqual(payload["location"]["camera_id"], "CAM-B")
        self.assertEqual(payload["images"], [])
        self.assertNotIn("local_path", str(payload))

    def _engine(self, settings, folder):
        return EventEngine(settings, load_topology(settings), EventOutbox(Path(folder) / "events.db"),
            EvidenceStore(Path(folder) / "evidence"), OpenAICompatibleVisionReviewer({"enabled": False}))

    @staticmethod
    def _obs(camera_id, at, persons=(), objects=(), healthy=True):
        return CameraObservation(camera_id, "CORRIDOR-01", at, 640, 360, True, healthy,
            persons=persons, objects=objects, baseline_version=1, baseline_state=BaselineState.FROZEN)

    def _prime(self, engine, at):
        for camera_id in ("CAM-A", "CAM-B", "CAM-C"):
            engine.process(self._obs(camera_id, at))


if __name__ == "__main__":
    unittest.main()
