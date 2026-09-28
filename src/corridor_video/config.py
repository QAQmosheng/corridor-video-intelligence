from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class Settings:
    source_path: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.values.get(name, {}))

    def path(self, key: str) -> Path:
        raw = self.values["paths"][key]
        value = Path(raw)
        if not value.is_absolute():
            value = (self.source_path.parent / value).resolve()
        return value

    def resolve(self, section: str, key: str) -> Path:
        value = Path(self.values[section][key])
        if not value.is_absolute():
            value = (self.source_path.parent / value).resolve()
        return value


def load_settings(path: str | Path) -> Settings:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    required = {"service", "paths", "baseline", "vision", "person", "session", "events", "cameras"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"配置缺少章节: {', '.join(missing)}")
    return Settings(source, values)
