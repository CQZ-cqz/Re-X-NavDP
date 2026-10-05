"""Track only controls actually applied by the simulator, not queued plans."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[2]
_REX_BASE = _REX_ROOT / "x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

from collections import deque
import numpy as np


class ExecutionHistory:
    def __init__(self, batch_size, max_segments=64):
        self.segments = [deque(maxlen=max_segments) for _ in range(batch_size)]

    def reset_env(self, i):
        self.segments[i].clear()

    def record(self, plan_id, dt, start_positions, start_quaternions,
               end_positions, end_quaternions, dones):
        if plan_id is None:
            return
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Control duration must be positive simulation seconds")
        for i, history in enumerate(self.segments):
            if bool(dones[i]):
                self.reset_env(i)
                continue
            if not history or history[-1]["plan_id"] != plan_id:
                history.append({"plan_id": int(plan_id), "duration_s": 0.0,
                    "start_position": np.asarray(start_positions[i]).tolist(),
                    "start_quaternion": np.asarray(start_quaternions[i]).tolist()})
            history[-1]["duration_s"] += float(dt)
            history[-1]["end_position"] = np.asarray(end_positions[i]).tolist()
            history[-1]["end_quaternion"] = np.asarray(end_quaternions[i]).tolist()

    def snapshot(self):
        # Values are replaced, never mutated in place after publication.
        return [[dict(segment) for segment in history] for history in self.segments]
