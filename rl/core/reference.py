"""Timestamped reference paths with pose alignment and bounded progress search."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(quaternion):
    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
        raise ValueError('Expected nonzero finite xyzw quaternion')
    return Rotation.from_quat(quat)


def project_to_path(world, arc, position, arc_min=None, arc_max=None):
    """Project a 3D position onto a planar world path (no tracker mutation).

    Returns ``(tangent, cross_track, arc_position)`` where tangent is the unit
    world tangent at the nearest segment and cross_track is the planar distance.
    """
    world = np.asarray(world, dtype=float)
    arc = np.asarray(arc, dtype=float)
    pos = np.asarray(position, dtype=float)[:2]
    if world.ndim != 2 or world.shape[0] < 2 or world.shape[1] < 2 or arc.shape != (world.shape[0],):
        raise ValueError('World path needs at least two planar waypoints and a matching arc')
    start, vector = world[:-1, :2], np.diff(world[:, :2], axis=0)
    segment_arc = np.diff(arc)
    norm2 = np.maximum((vector**2).sum(-1), 1e-12)
    frac = np.clip(((pos-start)*vector).sum(-1)/norm2, 0., 1.)
    # Reward settlement must use the same local progress neighbourhood as the
    # online tracker. Otherwise a loop or nearby parallel segment can make the
    # nearest-point projection jump to an unrelated part of the path.
    allowed = np.ones(len(vector), dtype=bool)
    if arc_min is not None:
        allowed &= arc[1:] >= float(arc_min)
        frac = np.maximum(frac, np.clip((float(arc_min)-arc[:-1])/
            np.maximum(segment_arc, 1e-12), 0., 1.))
    if arc_max is not None:
        allowed &= arc[:-1] <= float(arc_max)
        frac = np.minimum(frac, np.clip((float(arc_max)-arc[:-1])/
            np.maximum(segment_arc, 1e-12), 0., 1.))
    if not allowed.any():
        raise ValueError('Projection window does not overlap the path')
    arc_pos = arc[:-1] + frac*np.diff(arc)
    proj = start + vector*frac[:, None]
    dist = np.linalg.norm(proj-pos, axis=-1)
    idx = int(np.argmin(np.where(allowed, dist, np.inf)))
    tangent = vector[idx]/(np.linalg.norm(vector[idx]) + 1e-12)
    return tangent, float(dist[idx]), float(arc_pos[idx])


class ReferenceTracker:
    def __init__(self, points=8, max_age_s=3., lookahead_m=1.):
        if points < 1 or max_age_s <= 0 or lookahead_m <= 0:
            raise ValueError('Reference dimensions and limits must be positive')
        self.points, self.max_age_s, self.lookahead_m = points, max_age_s, lookahead_m
        self.reset(0)

    def reset(self, episode_id):
        self.episode_id = int(episode_id)
        self.world = None
        self.plan_id = -1
        self.progress = 0.
        self.last_now = None

    def publish(self, trajectory, position, quaternion, capture_time, plan_id, episode_id):
        if int(episode_id) != self.episode_id or int(plan_id) <= self.plan_id:
            return False
        path = np.asarray(trajectory, dtype=float)
        pos = np.asarray(position, dtype=float)
        if path.ndim != 2 or path.shape[0] < 1 or path.shape[1] != 3 or pos.shape != (3,):
            raise ValueError('Expected T x 3 local path and 3D capture position')
        if not np.isfinite(path).all() or not np.isfinite(pos).all() or not np.isfinite(capture_time):
            raise ValueError('Reference must be finite')
        if self.world is not None and capture_time < self.capture_time:
            return False
        path = np.concatenate((np.zeros((1, 3)), path.copy()))
        path[:, 2] = 0.  # Existing X-NavDP tracker uses planar waypoints.
        world = _rotation(quaternion).apply(path) + pos
        distance = np.linalg.norm(np.diff(world[:, :2], axis=0), axis=-1)
        keep = np.r_[True, distance > 1e-7]
        self.world = world[keep]
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(self.world[:, :2], axis=0), axis=-1))]
        self.capture_time = float(capture_time)
        self.capture_position = pos.copy()
        self.capture_quaternion = np.asarray(quaternion, dtype=float).copy()
        self.plan_id, self.progress = int(plan_id), 0.
        return True

    def sample(self, position, quaternion, now):
        if not np.isfinite(now) or (self.last_now is not None and now < self.last_now):
            raise ValueError('Time must be finite and monotonic; reset on new episode')
        self.last_now = float(now)
        output = np.zeros((self.points, 4), dtype=np.float32)
        mask = np.zeros(self.points, dtype=np.float32)
        age = 0. if self.world is None else now-self.capture_time
        if self.world is None or age < 0 or age > self.max_age_s:
            return output, mask, age, False
        pos = np.asarray(position, dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError('Current position must be finite XYZ')
        rot = _rotation(quaternion)
        if len(self.world) > 1:
            start, vector = self.world[:-1], np.diff(self.world, axis=0)
            norm2 = np.maximum((vector[:, :2]**2).sum(-1), 1e-12)
            fraction = np.clip(((pos-start)[:, :2]*vector[:, :2]).sum(-1)/norm2, 0, 1)
            candidates = self.arc[:-1] + fraction*np.diff(self.arc)
            projection = start + vector*fraction[:, None]
            # Prevent nearest-point jumps across distant loops/self-intersections.
            allowed = (candidates >= self.progress-.05) & (candidates <= self.progress+self.lookahead_m)
            scores = np.linalg.norm(projection[:, :2]-pos[:2], axis=-1)
            if allowed.any():
                index = np.argmin(np.where(allowed, scores, np.inf))
                if scores[index] > self.lookahead_m:
                    return output, mask, age, False
                self.progress = max(self.progress, float(candidates[index]))
            else:
                return output, mask, age, False
        elif np.linalg.norm(self.world[0, :2]-pos[:2]) > self.lookahead_m:
            return output, mask, age, False
        targets = self.progress + np.linspace(.05, self.lookahead_m, self.points)
        # Always expose the final goal point, including a stationary reference.
        mask[:] = targets <= self.arc[-1]
        mask[0] = 1
        targets = np.minimum(targets, self.arc[-1])
        world = np.stack([np.interp(targets, self.arc, self.world[:, j]) for j in range(3)], -1)
        output[:, :2] = rot.inv().apply(world-pos)[:, :2]
        if len(self.world) > 1:
            indices = np.clip(np.searchsorted(self.arc, targets, side='right')-1, 0, len(self.world)-2)
            tangent = self.world[indices+1]-self.world[indices]
            local = rot.inv().apply(tangent)[:, :2]
            output[:, 2:] = local/np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1e-9)
        output *= mask[:, None]
        return output, mask, age, True

    def reward_geometry(self):
        """Immutable world-path snapshot for reward settlement (does not advance progress).

        The reward for action a_t is settled against the OLD path captured when a_t
        was computed, even if a new plan arrives before the transition completes.
        """
        if self.world is None:
            return None
        return {'world': self.world.copy(), 'arc': self.arc.copy(),
                'progress': float(self.progress), 'lookahead_m': float(self.lookahead_m),
                'plan_id': self.plan_id, 'capture_time': self.capture_time}
