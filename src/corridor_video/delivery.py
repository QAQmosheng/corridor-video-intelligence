from __future__ import annotations

from typing import Any
import json
import os
import urllib.request


def to_weknora_payload(event: dict[str, Any]) -> dict[str, Any]:
    """Translate the internal event into the requirement 10.6 contract."""
    images = []
    for image in event.get("images", []):
        # Local paths are never exposed across the service boundary. The
        # uploader adds a controlled URL before delivery.
        if not image.get("url"):
            continue
        images.append({key: image[key] for key in ("image_id", "url", "media_type", "captured_at", "sha256") if key in image})
    return {
        "schema_version": event.get("schema_version", "1.0"),
        "event_id": event["event_id"],
        "event_text": event["event_text"],
        "event_type": event["event_type"],
        "event_time": event["captured_at"],
        "location": {
            "corridor_section_id": event["corridor_id"],
            "camera_id": event["camera_id"],
            "location_text": event.get("metadata", {}).get("location_text", ""),
        },
        "device_id": event["camera_id"],
        "algorithm_confidence": event["confidence"],
        "images": images,
        "upstream_metadata": {
            "source_system": "corridor-native-vision",
            "source_algorithm": "baseline_gate+native_semantic+centroid_tracker+corridor_fsm",
            "source_algorithm_version": "0.2.0",
            "corridor_session_id": event.get("session_id"),
            "detections": event.get("detections", []),
            "reason_code": event["reason_code"],
            "decision_metadata": event.get("metadata", {}),
        },
    }


class WeKnoraEventClient:
    """Prepared adapter for the future WeKnora POST /api/v1/events endpoint."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def send(self, event: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.get("enabled", False):
            return {"status": "disabled"}
        token = os.getenv(self.settings.get("api_key_env", "WEKNORA_EVENT_API_KEY"), "")
        payload = to_weknora_payload(event)
        request = urllib.request.Request(
            self.settings["endpoint"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8", "Authorization": f"Bearer {token}", "Idempotency-Key": payload["event_id"]},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=float(self.settings.get("timeout_seconds", 10))) as response:
            return json.loads(response.read().decode("utf-8"))
