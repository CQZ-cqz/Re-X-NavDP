"""Local failed-direction memory over acknowledged executed plans.

Trajectories are cumulative robot-frame XYZ positions in metres. Pose uses
world XYZ and SciPy xyzw quaternions. Only planar yaw is used (Dingo).
Execution duration is simulation time, never wall-clock inference latency.
This is an empirical recovery selector, not a collision/safety guarantee.
"""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[2]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class RecoveryConfig:
    history_size: int = 8
    min_execution_s: float = 0.5
    min_motion_m: float = 0.08
    min_turn_rad: float = 0.25
    locality_m: float = 0.75
    memory_ttl_s: float = 30.0
    escape_distance_m: float = 0.5
    min_hold_s: float = 1.0
    cooldown_s: float = 1.0
    failure_weight: float = 2.0
    persistence_weight: float = 0.25
    direction_waypoint: int = 5

    def __post_init__(self):
        positive = (self.history_size, self.min_execution_s, self.min_motion_m,
                    self.min_turn_rad, self.locality_m, self.memory_ttl_s,
                    self.escape_distance_m, self.min_hold_s, self.cooldown_s)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Recovery sizes, durations and distances must be positive")
        if not isinstance(self.history_size, int) or self.direction_waypoint < 0:
            raise ValueError("history_size must be an integer; waypoint must be nonnegative")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.failure_weight, self.persistence_weight)):
            raise ValueError("Recovery weights must be finite and nonnegative")


@dataclass
class _State:
    failures: deque
    pending: dict = field(default_factory=dict)
    mode: str = "NORMAL"
    anchor: object = None
    entered_at: float = 0.0
    cooldown_until: float = 0.0
    last_direction: object = None
    last_time: object = None
    episode: int = 0


def _yaw(quat):
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("Expected finite nonzero xyzw quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


class RecoverySelector:
    MODES = ("baseline", "q_only", "state_only", "memory")

    def __init__(self, mode="baseline", config=None):
        if mode not in self.MODES:
            raise ValueError(f"Unknown recovery mode: {mode}")
        self.mode = mode
        self.config = config or RecoveryConfig()
        self.states = []
        self._episode_counter = 0

    def reset(self, batch_size):
        self.states = [None] * int(batch_size)
        for i in range(int(batch_size)):
            self.reset_env(i)

    def reset_env(self, i):
        self._episode_counter += 1
        self.states[i] = _State(deque(maxlen=self.config.history_size),
                                episode=self._episode_counter)

    def select(self, i, trajectories, values, position, quaternion, stuck,
               plan_id=None, sim_time=None, executed_plan_id=None,
               executed_duration_s=0.0, executed_segments=None):
        cfg, state = self.config, self.states[i]
        paths = np.asarray(trajectories, dtype=float)
        q = np.asarray(values, dtype=float)
        if paths.ndim != 3 or paths.shape[1] == 0 or paths.shape[2] != 3:
            raise ValueError("Expected N x T x 3 cumulative trajectories")
        if q.shape != (paths.shape[0],) or not np.isfinite(q).all() or not np.isfinite(paths).all():
            raise ValueError("Candidates and Q scores must be finite and aligned")
        selected = int(np.argmax(q))
        report = {"mode": self.mode, "episode": state.episode, "plan_id": plan_id,
                  "stuck": bool(stuck), "state": state.mode, "selected": selected,
                  "q": q.tolist(), "feedback_available": False}
        if self.mode == "baseline":
            if stuck:
                selected = int(np.random.randint(0, len(q)))
            report["selected"] = selected
            return selected, report
        if self.mode == "q_only":
            return selected, report
        if position is None or quaternion is None or sim_time is None or plan_id is None:
            report["fallback"] = "missing_pose_or_execution_protocol"
            return selected, report

        now = float(sim_time)
        pos = np.asarray(position, dtype=float)[:2]
        if pos.shape != (2,) or not np.isfinite(pos).all() or not math.isfinite(now):
            raise ValueError("Expected finite position and simulation time")
        duration = float(executed_duration_s)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid executed duration")
        if state.last_time is not None and now < state.last_time:
            raise ValueError("Simulation time moved backwards; reset this environment")
        state.last_time = now
        yaw = _yaw(quaternion)
        report["feedback_available"] = True
        report["sim_time"] = now
        report["executed_plan_id"] = executed_plan_id
        report["executed_duration_s"] = duration
        state.failures = deque((f for f in state.failures if now-f[2] <= cfg.memory_ttl_s),
                               maxlen=cfg.history_size)

        # Client reports motion measured during actual application of this plan.
        # Repeated cumulative segments are safe: one vote per issued plan.
        for segment in executed_segments or []:
            attempt = state.pending.get(segment["plan_id"])
            if not math.isfinite(segment["duration_s"]) or segment["duration_s"] < 0:
                raise ValueError("Invalid execution segment duration")
            if attempt is None or segment["duration_s"] < cfg.min_execution_s:
                continue
            origin = np.asarray(segment["start_position"], dtype=float)[:2]
            endpoint = np.asarray(segment["end_position"], dtype=float)[:2]
            if not np.isfinite(origin).all() or not np.isfinite(endpoint).all():
                raise ValueError("Execution segment positions must be finite")
            start_yaw = _yaw(segment["start_quaternion"])
            end_yaw = _yaw(segment["end_quaternion"])
            displacement = float(np.linalg.norm(endpoint-origin))
            turn = abs(math.atan2(math.sin(end_yaw-start_yaw), math.cos(end_yaw-start_yaw)))
            failed = bool(stuck and displacement < cfg.min_motion_m and turn < cfg.min_turn_rad)
            if failed and np.linalg.norm(attempt["direction"]) > 0:
                state.failures.append((origin.copy(), attempt["direction"].copy(), now))
            report["observed_attempt"] = {"plan_id": segment["plan_id"],
                "displacement_m": displacement, "turn_rad": turn, "failed": failed}
            state.pending.pop(segment["plan_id"])
        state.pending = {k: a for k, a in state.pending.items() if now-a["created"] <= cfg.memory_ttl_s}

        if state.mode == "COOLDOWN" and now >= state.cooldown_until:
            state.mode = "NORMAL"
        if state.mode == "NORMAL" and stuck:
            state.mode, state.anchor, state.entered_at = "RECOVER", pos.copy(), now
        if (state.mode == "RECOVER" and not stuck
                and now-state.entered_at >= cfg.min_hold_s
                and np.linalg.norm(pos-state.anchor) >= cfg.escape_distance_m):
            state.mode = "COOLDOWN"
            state.cooldown_until = now + cfg.cooldown_s
            state.last_direction = None

        waypoint = min(cfg.direction_waypoint, paths.shape[1]-1)
        vectors = paths[:, waypoint, :2]
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(lengths[:, None], 1e-9)
        c, s = math.cos(yaw), math.sin(yaw)
        directions = directions @ np.array([[c, s], [-s, c]])
        costs = np.zeros(len(q))
        if self.mode == "memory":
            for origin, direction, timestamp in state.failures:
                distance = np.linalg.norm(pos-origin)
                if distance < cfg.locality_m:
                    similarity = np.clip(directions @ direction, 0, 1)**2
                    costs += similarity * (1-distance/cfg.locality_m) * math.exp(-(now-timestamp)/cfg.memory_ttl_s)
        normalized = (q-q.mean()) / max(float(q.std()), 1e-6)
        persistence = np.zeros(len(q))
        if state.last_direction is not None:
            persistence = np.clip(directions @ state.last_direction, -1, 1)
        scores = normalized - cfg.failure_weight*costs + cfg.persistence_weight*persistence
        if state.mode == "RECOVER":
            selected = int(np.argmax(scores))
            state.last_direction = directions[selected].copy()
        # No generic visit penalty: valid backtracking may traverse old positions.
        state.pending[int(plan_id)] = {"direction": directions[selected].copy(),
                                      "created": now}
        while len(state.pending) > 128:
            state.pending.pop(next(iter(state.pending)))
        report.update(state=state.mode, selected=selected, failure_count=len(state.failures),
                      failure_cost=costs.tolist(), recovery_scores=scores.tolist())
        return selected, report
