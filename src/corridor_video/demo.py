from __future__ import annotations

from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import urlparse
import json

import cv2
import numpy as np

from .config import load_settings
from .engine import EventEngine
from .evidence import EvidenceStore
from .outbox import EventOutbox
from .topology import load_topology
from .vision import NativeVisionPipeline
from .vlm import OpenAICompatibleVisionReviewer


ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "web"
CONFIG = ROOT / "config" / "default.toml"


class DemoRuntime:
    def __init__(self) -> None:
        self.lock = Lock()
        self._temp = TemporaryDirectory(prefix="corridor-native-demo-")
        self.reset()

    def reset(self) -> dict:
        settings = load_settings(CONFIG)
        topology = load_topology(settings.values)
        db = Path(self._temp.name) / "events.db"
        if db.exists():
            db.unlink()
        self.settings, self.topology = settings, topology
        self.vision = NativeVisionPipeline({"baseline": settings.section("baseline"), **settings.section("vision")})
        self.engine = EventEngine(settings.values, topology, EventOutbox(db), EvidenceStore(Path(self._temp.name) / "evidence"),
                                  OpenAICompatibleVisionReviewer(settings.section("vlm")))
        self.step_index = 0
        self.events: list[dict] = []
        return self.state([], {})

    def close(self) -> None:
        self._temp.cleanup()

    def step(self) -> dict:
        with self.lock:
            now = datetime(2026, 9, 10, 9, 0, tzinfo=timezone(timedelta(hours=8))) + timedelta(seconds=self.step_index)
            scene = self._scene(self.step_index)
            detections: dict[str, dict] = {}
            tick_events = []
            for camera_id in self.topology.corridors["CORRIDOR-01"]:
                frame = self._draw_frame(camera_id, scene.get(camera_id, []))
                session_active = "CORRIDOR-01" in self.engine.sessions
                if session_active:
                    self.vision.freeze(camera_id)
                else:
                    self.vision.unfreeze(camera_id)
                observation = self.vision.analyze(frame, camera_id, "CORRIDOR-01", now,
                                                  allow_baseline_update=not session_active)
                new_events = self.engine.process(observation)
                tick_events.extend(event.to_dict() for event in new_events)
                detections[camera_id] = {
                    "changed": observation.stable_change,
                    "changed_ratio": round(observation.changed_ratio, 4),
                    "baseline_version": observation.baseline_version,
                    "baseline_state": observation.baseline_state.value,
                    "persons": [self._person_json(item) for item in observation.persons],
                    "objects": [self._object_json(item) for item in observation.objects],
                    "fire": [self._candidate_json(item) for item in observation.fire_candidates],
                    "water": None if not observation.water_candidate else self._candidate_json(observation.water_candidate),
                }
            self.events.extend(tick_events)
            self.step_index += 1
            return self.state(tick_events, detections)

    def state(self, tick_events: list[dict], detections: dict) -> dict:
        phase, description = self._phase(self.step_index)
        return {
            "step": self.step_index,
            "phase": phase,
            "description": description,
            "complete": self.step_index >= 35,
            "corridor": self.engine.snapshot("CORRIDOR-01"),
            "detections": detections,
            "new_events": tick_events,
            "events": self.events[-12:],
            "metrics": {
                "frames": self.step_index * 3,
                "llm_calls": 0,
                "events": len(self.events),
                "algorithm": "Native CV 0.1",
            },
        }

    @staticmethod
    def _phase(step: int) -> tuple[str, str]:
        if step < 5:
            return "基准标定", "三台摄像头分别建立自己的稳定基准。"
        if step < 11:
            return "人员进入", "CAM-A 发现人员，冻结整个廊段的会话前基准；CAM-B 登记工具箱候选。"
        if step < 16:
            return "跨镜移动", "人员从 CAM-A 移动到 CAM-B，会话保持为同一个 corridor_session_id。"
        if step < 22:
            return "离场确认", "全廊段暂时无人，但仍处于盲区与离场等待窗口，不产生遗留物告警。"
        if step < 25:
            return "持续性验证", "工具箱在 CAM-B 相对冻结基准持续存在，累计独处时长。"
        if step < 32:
            return "生成事件", "三项条件同时满足，只在物品所在摄像头生成一次遗留物事件。"
        return "视觉复核", "CAM-C 发现火焰候选；外部视觉模型未启用，因此进入人工复核队列。"

    @staticmethod
    def _scene(step: int) -> dict[str, list[str]]:
        scene = {"CAM-A": [], "CAM-B": [], "CAM-C": []}
        if 5 <= step < 11:
            scene["CAM-A"].append("worker")
            scene["CAM-B"].append("toolbox")
        elif 11 <= step < 16:
            scene["CAM-B"].extend(["worker", "toolbox"])
        elif 16 <= step:
            scene["CAM-B"].append("toolbox")
        if 32 <= step < 35:
            scene["CAM-C"].append("fire")
        return scene

    @staticmethod
    def _draw_frame(camera_id: str, entities: list[str]) -> np.ndarray:
        frame = np.full((360, 640, 3), (35, 39, 44), dtype=np.uint8)
        cv2.line(frame, (0, 286), (640, 286), (85, 90, 95), 3)
        cv2.line(frame, (0, 80), (640, 80), (54, 59, 64), 2)
        if "worker" in entities:
            cv2.rectangle(frame, (180, 135), (230, 280), (190, 90, 45), -1)
            cv2.rectangle(frame, (180, 125), (230, 154), (0, 0, 230), -1)
        if "toolbox" in entities:
            cv2.rectangle(frame, (390, 238), (470, 288), (60, 190, 80), -1)
            cv2.rectangle(frame, (412, 225), (448, 242), (60, 190, 80), 6)
        if "fire" in entities:
            points = np.array([[520, 288], [500, 245], [527, 258], [538, 205], [558, 255], [580, 236], [574, 288]])
            cv2.fillPoly(frame, [points], (0, 105, 255))
        return frame

    @staticmethod
    def _person_json(item) -> dict:
        return {"track_id": item.track_id, "bbox": list(item.bbox.__dict__.values()), "helmet": item.helmet.value,
                "confidence": round(item.confidence, 3)}

    @staticmethod
    def _object_json(item) -> dict:
        return {"track_id": item.track_id, "bbox": list(item.bbox.__dict__.values()), "category": item.category,
                "confidence": round(item.confidence, 3)}

    @staticmethod
    def _candidate_json(item) -> dict:
        return {"bbox": None if item.bbox is None else list(item.bbox.__dict__.values()), "label": item.label,
                "confidence": round(item.confidence, 3)}


RUNTIME = DemoRuntime()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(RUNTIME.state([], {}))
        file = STATIC / ("index.html" if path == "/" else path.lstrip("/"))
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

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/step":
            return self._json(RUNTIME.step())
        if path == "/api/reset":
            with RUNTIME.lock:
                return self._json(RUNTIME.reset())
        self.send_error(404)

    def _json(self, value: dict) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    print(f"视频智能体动态原型: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
