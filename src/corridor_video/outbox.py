from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from contextlib import closing
import json
import sqlite3

from .models import VideoEvent


class EventOutbox:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS event_outbox (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        dedup_key TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_error TEXT
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON event_outbox(status, created_at)")

    def enqueue(self, event: VideoEvent, dedup_key: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with closing(self.connect()) as connection:
                with connection:
                    connection.execute(
                        "INSERT INTO event_outbox(event_id,event_type,dedup_key,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                        (event.event_id, event.event_type, dedup_key, json.dumps(event.to_dict(), ensure_ascii=False), now, now),
                    )
            return True
        except sqlite3.IntegrityError:
            return False

    def pending(self, limit: int = 100) -> list[dict]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT payload FROM event_outbox WHERE status='pending' ORDER BY created_at LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM event_outbox").fetchone()[0])

    def last_captured_at(self, dedup_key: str) -> datetime | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM event_outbox WHERE dedup_key=? ORDER BY created_at DESC LIMIT 1", (dedup_key,)
            ).fetchone()
        if row is None:
            return None
        return datetime.fromisoformat(json.loads(row["payload"])["captured_at"])
