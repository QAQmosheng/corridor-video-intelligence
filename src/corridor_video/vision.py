from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from typing import Any

import cv2
import numpy as np

from .baseline import DynamicBaseline
from .models import BaselineState, BoundingBox, CameraObservation, HelmetState, ObjectObservation, PersonObservation, VisualCandidate


@dataclass
class _Track:
    center: tuple[float, float]
    missed: int = 0


@dataclass
class CameraVisionState:
    baseline: DynamicBaseline
    tracks: dict[str, _Track] = field(default_factory=dict)
    next_track_id: int = 1


class NativeVisionPipeline:
    """Self-contained OpenCV prototype; it loads no external detector weights."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self._states: dict[str, CameraVisionState] = {}

    def freeze(self, camera_id: str) -> None:
        if camera_id in self._states:
            self._states[camera_id].baseline.freeze()

    def unfreeze(self, camera_id: str) -> None:
        if camera_id in self._states:
            self._states[camera_id].baseline.unfreeze()

    def invalidate(self, camera_id: str) -> None:
        """Discard the active reference before a reconnect or camera movement."""
        if camera_id in self._states:
            self._states[camera_id].baseline.invalidate()

    def analyze(self, frame: np.ndarray, camera_id: str, corridor_id: str, captured_at: datetime,
                healthy: bool = True, allow_baseline_update: bool = True) -> CameraObservation:
        state = self._states.setdefault(camera_id, CameraVisionState(self._new_baseline()))
        height, width = frame.shape[:2]
        if not healthy:
            state.baseline.invalidate()
            return CameraObservation(camera_id, corridor_id, captured_at, width, height, False, healthy=False)

        # A 320x180 reference is enough for people, but small low-contrast
        # objects (cardboard boxes in particular) lose their edges at that
        # resolution.  Keep the reference size configurable and use a larger
        # default while still doing the expensive semantic work only inside
        # the changed mask.
        baseline_width = int(self.settings.get("baseline_width", 640))
        baseline_height = int(self.settings.get("baseline_height", 360))
        small = cv2.resize(frame, (baseline_width, baseline_height), interpolation=cv2.INTER_AREA)
        change = state.baseline.process(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), allow_update=allow_baseline_update)
        persons: list[PersonObservation] = []
        objects: list[ObjectObservation] = []
        fire: list[VisualCandidate] = []
        water: VisualCandidate | None = None
        if change.changed:
            candidates = self.semantic_candidates(frame, change.changed_mask)
            boxes = [item[1] for item in candidates if item[0] in {"person", "object"}]
            track_ids = iter(self._assign_tracks(state, boxes))
            for label, box, confidence, attributes in candidates:
                if label == "person":
                    persons.append(PersonObservation(next(track_ids), box, confidence,
                        attributes.get("helmet", HelmetState.UNKNOWN), float(attributes.get("helmet_confidence", 0.0))))
                elif label == "object":
                    objects.append(ObjectObservation(next(track_ids), attributes.get("category", "object"), box, confidence, True))
                elif label in {"fire", "smoke"}:
                    fire.append(VisualCandidate(label, box, confidence))
                elif label == "waterlogging":
                    water = VisualCandidate(label, box, confidence)
        return CameraObservation(
            camera_id, corridor_id, captured_at, width, height, change.changed, healthy,
            tuple(persons), tuple(objects), tuple(fire), water,
            baseline_version=change.baseline_version, baseline_state=state.baseline.state,
            changed_ratio=change.changed_ratio, global_change=change.global_change,
        )

    def semantic_candidates(
        self, frame: np.ndarray, changed_mask: np.ndarray | None = None
    ) -> list[tuple[str, BoundingBox, float, dict[str, Any]]]:
        """Interpretable color/geometry candidates used to prove the new pipeline.

        This deterministic stage is deliberately replaceable by a newly trained
        ONNX detector later; baseline, tracking and event decisions remain unchanged.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        if changed_mask is None:
            saturation, value = hsv[:, :, 1], hsv[:, :, 2]
            mask = np.where((saturation > 65) & (value > 45), 255, 0).astype(np.uint8)
        else:
            # Difference gating must also localize inference. Upscale the mask
            # generated at baseline resolution and ignore timestamp overlays.
            mask = cv2.resize(changed_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            ignore_top = int(height * float(self.settings.get("ignore_top_ratio", 0.10)))
            if ignore_top > 0:
                mask[:ignore_top, :] = 0
        raw_change_mask = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilation = max(0, int(self.settings.get("change_mask_dilate_iterations", 2)))
        if dilation:
            mask = cv2.dilate(mask, kernel, iterations=dilation)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = max(120.0, width * height * float(self.settings.get("min_component_ratio", 0.001)))
        results: list[tuple[str, BoundingBox, float, dict[str, Any]]] = []
        helmet_centers: list[tuple[float, float]] = []

        # In this installation helmets are red. Use each compact red component
        # as an anchor to split a merged crowd mask into individual person boxes.
        red_mask = np.where(
            (mask > 0)
            & ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
            & (hsv[:, :, 1] > 100)
            & (hsv[:, :, 2] > 90),
            255, 0,
        ).astype(np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        helmet_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_helmet = width * height * float(self.settings.get("helmet_min_area_ratio", 0.000015))
        max_helmet = width * height * float(self.settings.get("helmet_max_area_ratio", 0.004))
        for helmet_contour in helmet_contours:
            helmet_area = cv2.contourArea(helmet_contour)
            if not min_helmet <= helmet_area <= max_helmet:
                continue
            hx, hy, hw, hh = cv2.boundingRect(helmet_contour)
            helmet_aspect = hw / max(hh, 1)
            if not 0.40 <= helmet_aspect <= 4.0:
                continue
            if hy + hh / 2 >= height * float(self.settings.get("helmet_anchor_max_y_ratio", 0.92)):
                continue
            helmet_centers.append((hx + hw / 2, hy + hh / 2))
            person_width = max(float(hw) * float(self.settings.get("helmet_person_width_scale", 3.0)), width * 0.035)
            person_height = max(float(hh) * float(self.settings.get("helmet_person_height_scale", 9.0)), height * 0.16)
            center_x = hx + hw / 2
            person_box = BoundingBox(
                max(0.0, center_x - person_width / 2), float(hy),
                min(float(width), center_x + person_width / 2), min(float(height), hy + person_height),
            )
            results.append(("person", person_box, 0.82, {
                "helmet": HelmetState.WEARING, "helmet_confidence": 0.90,
            }))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if any(x <= center_x <= x + w and y <= center_y <= y + h
                   for center_x, center_y in helmet_centers):
                continue
            box = BoundingBox(float(x), float(y), float(x + w), float(y + h))
            roi = hsv[y:y + h, x:x + w]
            hue, sat, val = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
            component = mask[y:y + h, x:x + w] > 0
            component_pixels = max(1, int(np.count_nonzero(component)))
            orange = float(np.count_nonzero(component & ((hue < 25) | (hue > 170)) & (sat > 120) & (val > 130)) / component_pixels)
            blue = float(np.count_nonzero(component & (hue > 85) & (hue < 135) & (sat > 70)) / component_pixels)
            red_hint = float(np.count_nonzero(
                component & ((hue <= 10) | (hue >= 170)) & (sat > 100) & (val > 90)
            ) / component_pixels)
            material_hint = float(np.count_nonzero(
                component & (sat > 35) & (val > 35)
            ) / component_pixels)
            aspect = h / max(w, 1)
            if orange > 0.42 and aspect < 1.45:
                results.append(("fire", box, min(0.98, 0.55 + orange * 0.4), {}))
            elif blue > 0.55 and y > height * 0.55 and w > h * 1.5:
                results.append(("waterlogging", box, min(0.95, 0.5 + blue * 0.4), {}))
            elif (
                aspect >= float(self.settings.get("person_min_aspect", 1.45))
                or (
                    h >= height * float(self.settings.get("person_group_min_height_ratio", 0.18))
                    and red_hint >= float(self.settings.get("person_group_red_hint_ratio", 0.008))
                    and aspect >= 0.75
                )
            ) and area >= width * height * 0.006:
                top = roi[:max(1, int(h * 0.24))]
                # Red wraps around both ends of OpenCV's HSV hue range.
                red = float(np.mean(
                    ((top[:, :, 0] <= 10) | (top[:, :, 0] >= 170))
                    & (top[:, :, 1] > 100)
                    & (top[:, :, 2] > 90)
                ))
                helmet_threshold = float(self.settings.get("red_helmet_ratio", 0.12))
                # A wide merged contour can contain several people. The color
                # rule cannot assign helmets to individuals, so it must remain
                # unknown instead of creating a false PPE alarm.
                merged_group = w / max(h, 1) > float(self.settings.get("person_group_width_height_ratio", 0.68))
                helmet = (
                    HelmetState.WEARING
                    if not merged_group and red >= helmet_threshold
                    else HelmetState.UNKNOWN
                )
                # Geometry alone is not enough to assert that a changed
                # vertical region is a person: rails, reflections and pipe
                # edges have the same silhouette in a perspective scene.
                # Keep unknown regions internal unless explicitly enabled for
                # debugging; production confirmation belongs to ONNX/VLM.
                if helmet != HelmetState.UNKNOWN or self.settings.get("emit_unknown_person_candidates", False):
                    results.append(("person", box, min(0.96, 0.68 + area / (width * height)), {
                        "helmet": helmet,
                        "helmet_confidence": min(0.98, 0.65 + abs(red - helmet_threshold)),
                    }))
            elif (
                y + h / 2 >= height * float(self.settings.get("portable_min_center_y_ratio", 0.52))
                and area <= width * height * float(self.settings.get("portable_max_area_ratio", 0.08))
                and 0.25 <= aspect <= 3.0
                and blue <= float(self.settings.get("portable_max_blue_ratio", 0.30))
                and red_hint <= float(self.settings.get("portable_max_red_ratio", 0.25))
                and material_hint >= float(self.settings.get("portable_min_material_ratio", 0.22))
                and not (h >= height * 0.15 and red_hint >= 0.005)
            ):
                results.append(("object", box, 0.76, {"category": "portable_object"}))
            # Unknown changed regions are intentionally ignored. Treating every
            # unmatched contour as a portable object caused pipes, lamps and
            # timestamp overlays to be boxed as false alarms.
        # A box can be connected to a long floor/rail change in the binary
        # difference mask.  The parent contour then looks person-shaped and
        # used to be discarded as an unknown vertical region.  Split compact
        # brown material components out of that parent contour so cardboard
        # boxes remain independently observable after the person walks away.
        brown = np.where(
            (raw_change_mask > 0)
            & (hsv[:, :, 0] >= int(self.settings.get("box_hue_min", 10)))
            & (hsv[:, :, 0] <= int(self.settings.get("box_hue_max", 32)))
            & (hsv[:, :, 1] >= int(self.settings.get("box_saturation_min", 35)))
            & (hsv[:, :, 2] >= 40) & (hsv[:, :, 2] <= 235),
            255, 0,
        ).astype(np.uint8)
        # A slightly stronger opening breaks the thin brown/yellow line that
        # otherwise connects the carton to the cable-rack edge.
        brown_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        brown = cv2.morphologyEx(brown, cv2.MORPH_OPEN, brown_kernel)
        brown_contours, _ = cv2.findContours(brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box_minimum = width * height * float(self.settings.get("box_min_area_ratio", 0.00025))
        box_maximum = width * height * float(self.settings.get("box_max_area_ratio", 0.012))
        for contour in brown_contours:
            area = cv2.contourArea(contour)
            if not box_minimum <= area <= box_maximum:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = h / max(w, 1)
            if not 0.45 <= aspect <= float(self.settings.get("box_max_aspect", 2.4)):
                continue
            if y + h / 2 < height * float(self.settings.get("portable_min_center_y_ratio", 0.52)):
                continue
            if w < width * 0.012 or h < height * 0.025:
                continue
            box = BoundingBox(float(x), float(y), float(x + w), float(y + h))
            if any(item[0] == "object" and self._box_iou(item[1], box) >= 0.30 for item in results):
                continue
            results.append(("object", box, 0.84, {"category": "box"}))
        return sorted(results, key=lambda item: (item[1].x1, item[1].y1))

    @staticmethod
    def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = left.width * left.height + right.width * right.height - intersection
        return intersection / union if union > 0 else 0.0

    def _new_baseline(self) -> DynamicBaseline:
        cfg = self.settings["baseline"]
        keys = ("pixel_threshold", "changed_area_ratio", "global_change_ratio", "stable_confirm_frames", "clear_confirm_frames", "learning_rate")
        return DynamicBaseline(**{key: cfg[key] for key in keys})

    @staticmethod
    def _assign_tracks(state: CameraVisionState, boxes: list[BoundingBox]) -> list[str]:
        assigned: list[str] = []
        available = set(state.tracks)
        for box in boxes:
            center = box.center
            choices = [(hypot(center[0] - state.tracks[key].center[0], center[1] - state.tracks[key].center[1]), key) for key in available]
            distance, best_id = min(choices, default=(1e9, ""))
            if distance >= 90:
                best_id = f"T{state.next_track_id:04d}"
                state.next_track_id += 1
                state.tracks[best_id] = _Track(center)
            else:
                state.tracks[best_id].center = center
                state.tracks[best_id].missed = 0
                available.remove(best_id)
            assigned.append(best_id)
        for track_id in available:
            state.tracks[track_id].missed += 1
        state.tracks = {key: value for key, value in state.tracks.items() if value.missed <= 8}
        return assigned
