from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
from copy import deepcopy
import json

from .config import load_settings
from .engine import EventEngine
from .evidence import EvidenceStore
from .outbox import EventOutbox
from .runner import monitor_rtsp, run_video_smoke
from .topology import Camera, Topology, load_topology
from .vision import NativeVisionPipeline
from .vlm import OpenAICompatibleVisionReviewer

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.toml"


def build_engine(settings):
    topology = load_topology(settings.values)
    return EventEngine(settings.values, topology, EventOutbox(settings.path("database")),
        EvidenceStore(settings.path("evidence_dir")), OpenAICompatibleVisionReviewer(settings.section("vlm")))


def build_standalone_engine(settings, camera_id: str, corridor_id: str):
    values = deepcopy(settings.values)
    topology = Topology(
        cameras={camera_id: Camera(corridor_id, camera_id, camera_id, "", 1, corridor_id)},
        corridors={corridor_id: [camera_id]},
    )
    return EventEngine(values, topology, EventOutbox(settings.path("database")),
        EvidenceStore(settings.path("evidence_dir")), OpenAICompatibleVisionReviewer(settings.section("vlm")))


def analyze_video(settings, source: str, camera_id: str, max_frames: int) -> dict:
    import cv2
    topology = load_topology(settings.values)
    if camera_id not in topology.cameras:
        raise ValueError(f"未知摄像头: {camera_id}")
    camera = topology.cameras[camera_id]
    engine = build_engine(settings)
    pipeline = NativeVisionPipeline({"baseline": settings.section("baseline"), **settings.section("vision")})
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    start = datetime.now(timezone(timedelta(hours=8)))
    frames = changed = detections = 0
    events: list[dict] = []
    try:
        while max_frames <= 0 or frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            now = start + timedelta(seconds=frames / fps)
            active = camera.corridor_id in engine.sessions
            pipeline.freeze(camera_id) if active else pipeline.unfreeze(camera_id)
            observation = pipeline.analyze(frame, camera_id, camera.corridor_id, now, allow_baseline_update=not active)
            frames += 1
            changed += int(observation.stable_change)
            detections += len(observation.persons) + len(observation.objects) + len(observation.fire_candidates) + int(bool(observation.water_candidate))
            events.extend(event.to_dict() for event in engine.process(observation))
    finally:
        capture.release()
    return {"source": source, "camera_id": camera_id, "frames": frames, "stable_change_frames": changed,
            "detections": detections, "events": events, "outbox": str(engine.outbox.database)}


def main() -> None:
    parser = argparse.ArgumentParser(description="独立管廊视频识别与廊段事件引擎")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")
    commands.add_parser("simulate")
    demo = commands.add_parser("demo")
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    smoke = commands.add_parser("video-smoke")
    smoke.add_argument("--source", required=True)
    smoke.add_argument("--max-frames", type=int, default=300)
    analyze = commands.add_parser("analyze-video")
    analyze.add_argument("--source", required=True)
    analyze.add_argument("--camera-id", default="CAM-A")
    analyze.add_argument("--max-frames", type=int, default=1000)
    rtsp = commands.add_parser("monitor-rtsp", help="监控单路 RTSP；断流后重采样基准")
    rtsp.add_argument("--source", required=True, help="rtsp://user:password@host/path")
    rtsp.add_argument("--camera-id", default="CAM-01")
    rtsp.add_argument("--corridor-id", default="CORRIDOR-01")
    rtsp.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行")
    rtsp.add_argument("--reconnect-delay", type=float, default=2.0)
    rtsp.add_argument("--max-reconnects", type=int, default=0, help="0 表示持续重试")
    live = commands.add_parser("live", help="启动带标注视频和报警列表的持续检测页面")
    live.add_argument("--source", required=True, help="MP4 文件或 RTSP 地址")
    live.add_argument("--camera-id", default="CAM-01")
    live.add_argument("--corridor-id", default="CORRIDOR-01")
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=8766)
    live.add_argument("--no-loop", action="store_true", help="文件播放结束后停止，不循环")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.command == "validate-config":
        topology = load_topology(settings.values)
        output = {"status": "ok", "module": settings.values["service"]["module_id"],
                  "cameras": len(topology.cameras), "corridors": len(topology.corridors),
                  "algorithm_dependency": "none (native OpenCV prototype)"}
    elif args.command == "simulate":
        from .demo import DemoRuntime
        runtime = DemoRuntime()
        result = None
        for _ in range(35):
            result = runtime.step()
        output = {"status": "ok", "steps": result["step"], "events": result["events"]}
    elif args.command == "demo":
        from .demo import serve
        return serve(args.host, args.port)
    elif args.command == "video-smoke":
        output = asdict(run_video_smoke(args.source, settings.section("baseline"), args.max_frames))
    elif args.command == "monitor-rtsp":
        pipeline = NativeVisionPipeline({"baseline": settings.section("baseline"), **settings.section("vision")})
        engine = build_standalone_engine(settings, args.camera_id, args.corridor_id)

        def print_event(event):
            print(json.dumps({"kind": "event", "event": event.to_dict()}, ensure_ascii=False), flush=True)

        output = asdict(monitor_rtsp(
            args.source, args.camera_id, args.corridor_id, pipeline, engine,
            max_frames=args.max_frames, reconnect_delay_seconds=args.reconnect_delay,
            max_reconnects=args.max_reconnects, on_event=print_event,
        ))
    elif args.command == "live":
        from .live import serve_live
        return serve_live(settings, args.source, args.camera_id, args.corridor_id,
                          args.host, args.port, loop_file=not args.no_loop)
    else:
        output = analyze_video(settings, args.source, args.camera_id, args.max_frames)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
