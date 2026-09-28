from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Camera:
    corridor_id: str
    camera_id: str
    name: str
    source: str
    order: int
    location: str = ""


@dataclass
class Topology:
    cameras: dict[str, Camera]
    corridors: dict[str, list[str]]

    def neighbors(self, camera_id: str) -> tuple[str, ...]:
        camera = self.cameras[camera_id]
        ordered = self.corridors[camera.corridor_id]
        index = ordered.index(camera_id)
        return tuple(ordered[max(0, index - 1):index] + ordered[index + 1:index + 2])


def load_topology(values: dict[str, Any]) -> Topology:
    cameras: dict[str, Camera] = {}
    corridors: dict[str, list[str]] = {}
    for row in values.get("cameras", []):
        camera = Camera(str(row["corridor_id"]), str(row["camera_id"]),
                        str(row.get("name", row["camera_id"])), str(row.get("source", "")),
                        int(row.get("order", 0)), str(row.get("location", "")))
        if camera.camera_id in cameras:
            raise ValueError(f"摄像头编号重复: {camera.camera_id}")
        cameras[camera.camera_id] = camera
        corridors.setdefault(camera.corridor_id, []).append(camera.camera_id)
    if not cameras:
        raise ValueError("配置中没有 [[cameras]]")
    for camera_ids in corridors.values():
        camera_ids.sort(key=lambda camera_id: cameras[camera_id].order)
    return Topology(cameras, corridors)
