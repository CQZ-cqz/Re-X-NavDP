"""Time-window displacement detector independent of planning call count."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[2]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

from collections import deque
import math
import numpy as np


class TimedStuckDetector:
    def __init__(self, window_s=2.0, distance_m=0.25):
        if not all(math.isfinite(v) and v > 0 for v in (window_s, distance_m)):
            raise ValueError("stuck window and distance must be positive and finite")
        self.window_s, self.distance_m = window_s, distance_m
        self.history = deque()
        self.clock = None

    def reset(self):
        self.history.clear()
        self.clock = None

    def update(self, xy, timestamp, clock="sim"):
        xy = np.asarray(xy, dtype=float)[:2]
        if timestamp is None or not math.isfinite(float(timestamp)) or not np.isfinite(xy).all():
            self.reset()
            return False, {"ready": False, "reason": "missing_or_invalid_time_or_pose"}
        now = float(timestamp)
        h = self.history
        if self.clock != clock or (h and (now < h[-1][0] or now-h[-1][0] > self.window_s)):
            self.reset()
        self.clock = clock
        # Multiple planning calls on the same observation do not advance time.
        if not h or now > h[-1][0]:
            h.append((now, xy.copy()))
        cutoff = now - self.window_s
        while len(h) > 1 and h[1][0] <= cutoff:
            h.popleft()
        report = {"clock": clock, "window_s": self.window_s,
                  "distance_threshold_m": self.distance_m, "span_s": now-h[0][0]}
        if len(h) < 2 or h[0][0] > cutoff + 1e-9:
            return False, dict(report, ready=False, reason="warming_up")
        # Interpolate the window boundary rather than including an older pose.
        fraction = (cutoff-h[0][0]) / (h[1][0]-h[0][0])
        boundary = h[0][1] + fraction * (h[1][1]-h[0][1])
        points = np.stack([boundary] + [entry[1] for entry in list(h)[1:]])
        displacement = float(np.linalg.norm(points-h[-1][1], axis=1).max())
        return displacement < self.distance_m, dict(report, ready=True, max_displacement_m=displacement)
