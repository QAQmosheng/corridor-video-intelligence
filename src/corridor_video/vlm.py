from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import base64
import json
import mimetypes
import os
import urllib.request


@dataclass(frozen=True)
class ReviewResult:
    status: str
    label: str
    confidence: float
    evidence: str = ""
    uncertainty: str = ""
    raw: dict[str, Any] | None = None


class OpenAICompatibleVisionReviewer:
    """Generic OpenAI-compatible vision endpoint with strict JSON output."""

    ALLOWED = {
        "fire_smoke_review": {"fire", "smoke", "none", "unknown"},
        "waterlogging_review": {"waterlogging", "none", "unknown"},
    }

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def review(self, task_type: str, image_paths: list[str], detector_hints: list[dict]) -> ReviewResult:
        if not self.settings.get("enabled", False):
            return ReviewResult("unavailable", "unknown", 0.0, uncertainty="VLM未启用")
        allowed = self.ALLOWED[task_type]
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"任务={task_type}。仅判断可见事实；标签只能为 {sorted(allowed)}。"
                "不要推测积水深度。返回JSON字段 label,confidence,evidence,uncertainty。"
                f"检测器提示={json.dumps(detector_hints, ensure_ascii=False)}"
            ),
        }]
        for image_path in image_paths[:4]:
            raw = Path(image_path).read_bytes()
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            data = base64.b64encode(raw).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
        payload = {
            "model": self.settings["model"],
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        api_key = os.getenv(str(self.settings.get("api_key_env", "VIDEO_VLM_API_KEY")), "")
        request = urllib.request.Request(
            self.settings["endpoint"],
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.settings.get("timeout_seconds", 15))) as response:
                body = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(body["choices"][0]["message"]["content"])
            label = str(parsed.get("label", "unknown"))
            if label not in allowed:
                label = "unknown"
            return ReviewResult(
                "succeeded", label, max(0.0, min(1.0, float(parsed.get("confidence", 0)))),
                str(parsed.get("evidence", "")), str(parsed.get("uncertainty", "")), body,
            )
        except Exception as exc:
            return ReviewResult("failed", "unknown", 0.0, uncertainty=str(exc))
