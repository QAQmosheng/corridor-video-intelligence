from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import shutil


class EvidenceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def preserve(self, source: str | None, event_id: str, captured_at: datetime, detections: list[dict] | None = None) -> list[dict]:
        if not source:
            return []
        source_path = Path(source)
        if not source_path.is_file():
            return []
        folder = self.root / captured_at.strftime("%Y/%m/%d")
        folder.mkdir(parents=True, exist_ok=True)
        suffix = source_path.suffix.lower() or ".jpg"
        destination = folder / f"{event_id}-original{suffix}"
        shutil.copy2(source_path, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        images = [{"kind": "original", "local_path": str(destination), "sha256": digest}]
        if detections:
            try:
                import cv2

                annotated = cv2.imread(str(source_path))
                if annotated is not None:
                    for item in detections:
                        box = item.get("bbox_xyxy")
                        if not box:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"{item.get('label', 'target')} {float(item.get('confidence', 0)):.2f}"
                        cv2.putText(annotated, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    annotated_path = folder / f"{event_id}-annotated.jpg"
                    cv2.imwrite(str(annotated_path), annotated)
                    images.append({
                        "kind": "annotated", "local_path": str(annotated_path),
                        "sha256": hashlib.sha256(annotated_path.read_bytes()).hexdigest(),
                    })
            except ImportError:
                pass
        return images
