from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from .baseline import DynamicBaseline
from .engine import EventEngine
from .models import CameraObservation, VideoEvent
from .vision import NativeVisionPipeline


@dataclass(frozen=True)
class SmokeResult:
    source: str
    frames_read: int
    changed_frames: int
    global_change_frames: int
    final_baseline_state: str
    baseline_version: int


@dataclass(frozen=True)
class StreamResult:
    source: str
    camera_id: str
    corridor_id: str
    frames_read: int
    stable_change_frames: int
    events_emitted: int
    reconnects: int
    final_baseline_state: str
    baseline_version: int


def safe_source_name(source: str) -> str:
    """Return a log-safe source name without RTSP credentials."""
    parts = urlsplit(source)
    if parts.scheme.lower() not in {"rtsp", "rtsps", "http", "https"}:
        return str(Path(source).name) if source else ""
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{host}{port}", parts.path, "", ""))


def monitor_rtsp(
    source: str,
    camera_id: str,
    corridor_id: str,
    pipeline: NativeVisionPipeline,
    engine: EventEngine,
    max_frames: int = 0,
    reconnect_delay_seconds: float = 2.0,
    max_reconnects: int = 0,
    on_event: Callable[[VideoEvent], None] | None = None,
) -> StreamResult:
    """Monitor one RTSP stream with reconnect-safe baseline handling.

    ``max_frames=0`` and ``max_reconnects=0`` mean unlimited. The camera is
    treated as a standalone corridor by the CLI so the complete session logic
    remains usable in a one-camera deployment.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("RTSP 监控需要安装 opencv-python-headless") from exc

    frames = changed = emitted = reconnects = 0
    last_observation: CameraObservation | None = None
    capture = None
    while max_frames <= 0 or frames < max_frames:
        if capture is None:
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                capture.release()
                capture = None
                reconnects += 1
                if max_reconnects > 0 and reconnects > max_reconnects:
                    break
                sleep(max(0.0, reconnect_delay_seconds))
                continue
        ok, frame = capture.read()
        if not ok:
            capture.release()
            capture = None
            pipeline.invalidate(camera_id)
            now = datetime.now(timezone.utc)
            engine.process(CameraObservation(camera_id, corridor_id, now, 0, 0, False, healthy=False))
            reconnects += 1
            if max_reconnects > 0 and reconnects > max_reconnects:
                break
            sleep(max(0.0, reconnect_delay_seconds))
            continue

        now = datetime.now(timezone.utc)
        session_active = corridor_id in engine.sessions
        pipeline.freeze(camera_id) if session_active else pipeline.unfreeze(camera_id)
        observation = pipeline.analyze(
            frame, camera_id, corridor_id, now,
            allow_baseline_update=not session_active,
        )
        last_observation = observation
        frames += 1
        changed += int(observation.stable_change)
        for event in engine.process(observation):
            emitted += 1
            if on_event:
                on_event(event)

    if capture is not None:
        capture.release()
    return StreamResult(
        safe_source_name(source), camera_id, corridor_id, frames, changed,
        emitted, reconnects,
        last_observation.baseline_state.value if last_observation else "uninitialized",
        last_observation.baseline_version if last_observation else 0,
    )


def run_video_smoke(source: str, baseline_config: dict[str, Any], max_frames: int = 300) -> SmokeResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("视频测试需要安装 opencv-python-headless") from exc

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    gate = DynamicBaseline(
        pixel_threshold=int(baseline_config["pixel_threshold"]),
        changed_area_ratio=float(baseline_config["changed_area_ratio"]),
        global_change_ratio=float(baseline_config["global_change_ratio"]),
        stable_confirm_frames=int(baseline_config["stable_confirm_frames"]),
        clear_confirm_frames=int(baseline_config["clear_confirm_frames"]),
        learning_rate=float(baseline_config["learning_rate"]),
    )
    frames = changed = global_changes = 0
    while frames < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        result = gate.process(gray)
        frames += 1
        changed += int(result.changed)
        global_changes += int(result.global_change)
    capture.release()
    return SmokeResult(source, frames, changed, global_changes, gate.state.value, gate.version)
