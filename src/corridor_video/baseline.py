from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .models import BaselineState


@dataclass(frozen=True)
class ChangeResult:
    changed: bool
    changed_ratio: float
    global_change: bool
    baseline_version: int
    changed_mask: np.ndarray


class DynamicBaseline:
    """Per-camera background model used only as a downstream inference gate."""

    def __init__(
        self,
        pixel_threshold: int = 25,
        changed_area_ratio: float = 0.015,
        global_change_ratio: float = 0.35,
        stable_confirm_frames: int = 3,
        clear_confirm_frames: int = 5,
        learning_rate: float = 0.02,
    ) -> None:
        self.pixel_threshold = pixel_threshold
        self.changed_area_ratio = changed_area_ratio
        self.global_change_ratio = global_change_ratio
        self.stable_confirm_frames = stable_confirm_frames
        self.clear_confirm_frames = clear_confirm_frames
        self.learning_rate = learning_rate
        self.state = BaselineState.UNINITIALIZED
        self.version = 0
        self._baseline: np.ndarray | None = None
        self._change_streak = 0
        self._clear_streak = 0

    def freeze(self) -> None:
        if self._baseline is not None:
            self.state = BaselineState.FROZEN

    def unfreeze(self) -> None:
        if self._baseline is not None:
            self.state = BaselineState.VALID

    def invalidate(self) -> None:
        self.state = BaselineState.INVALID

    def process(self, gray_frame: np.ndarray, allow_update: bool = True) -> ChangeResult:
        frame = gray_frame.astype(np.float32, copy=False)
        if self._baseline is None or self.state == BaselineState.INVALID:
            self._baseline = frame.copy()
            self.version += 1
            self.state = BaselineState.CALIBRATING
            self._clear_streak = 1
            return ChangeResult(False, 0.0, False, self.version, np.zeros(frame.shape, dtype=np.uint8))

        difference = np.abs(frame - self._baseline)
        changed_mask = np.where(difference >= self.pixel_threshold, 255, 0).astype(np.uint8)
        ratio = float(np.mean(changed_mask > 0))
        global_change = ratio >= self.global_change_ratio
        raw_change = self.changed_area_ratio <= ratio < self.global_change_ratio

        self._change_streak = self._change_streak + 1 if raw_change else 0
        self._clear_streak = self._clear_streak + 1 if not raw_change and not global_change else 0
        changed = self._change_streak >= self.stable_confirm_frames

        if self.state == BaselineState.CALIBRATING and self._clear_streak >= self.clear_confirm_frames:
            self.state = BaselineState.VALID

        can_learn = allow_update and self.state in {BaselineState.CALIBRATING, BaselineState.VALID}
        # Never absorb a candidate change while it is waiting for consecutive
        # frame confirmation. Otherwise a slow/stationary object can fade into
        # the reference before reaching ``stable_confirm_frames``.
        if can_learn and not raw_change and not global_change:
            self._baseline = (1.0 - self.learning_rate) * self._baseline + self.learning_rate * frame

        return ChangeResult(changed, ratio, global_change, self.version, changed_mask)
