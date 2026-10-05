"""Run from x-navdp: python -m unittest discover -s tests -p 'test_recovery*.py'."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import unittest
import numpy as np
from bridge.recovery import RecoveryConfig, RecoverySelector
from bridge.recovery.execution import ExecutionHistory


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.paths = np.zeros((3, 24, 3))
        for i, direction in enumerate(([1, 0], [-1, 0], [0, 1])):
            self.paths[i, :, :2] = np.arange(1, 25)[:, None] / 24 * np.array(direction)
        self.q = np.array([1., .9, .8])
        self.origin = [0., 0., 0.]
        self.quat = [0., 0., 0., 1.]

    def selector(self, mode="memory"):
        selector = RecoverySelector(mode)
        selector.reset(2)
        return selector

    def call(self, selector, t, plan, stuck=True, segments=None, pos=None, quat=None, env=0):
        return selector.select(env, self.paths, self.q, pos or self.origin,
            quat or self.quat, stuck, plan_id=plan, sim_time=t,
            executed_segments=segments)

    def segment(self, plan=1, end=None, duration=1., end_quat=None):
        return {"plan_id": plan, "duration_s": duration,
            "start_position": self.origin, "end_position": end or self.origin,
            "start_quaternion": self.quat, "end_quaternion": end_quat or self.quat}

    def test_baseline_reproduces_numpy_random(self):
        selector = self.selector("baseline")
        np.random.seed(123)
        expected = [np.random.randint(0, 3) for _ in range(10)]
        np.random.seed(123)
        actual = [self.call(selector, t, t)[0] for t in range(10)]
        self.assertEqual(actual, expected)
        self.assertEqual(self.call(selector, 11, 11, stuck=False)[0], 0)

    def test_memory_avoids_acknowledged_failed_direction(self):
        selector = self.selector()
        self.call(selector, 0, 1)
        selected, report = self.call(selector, 1, 2, segments=[self.segment()])
        self.assertEqual(selected, 1)  # Backtracking remains allowed.
        self.assertEqual(report["failure_count"], 1)
        self.assertGreater(report["failure_cost"][0], 0)
        self.assertEqual(report["failure_cost"][1], 0)

    def test_unexecuted_short_successful_or_turning_plans_not_failed(self):
        for segments in ([], [self.segment(plan=999)], [self.segment(duration=.1)],
                         [self.segment(end=[.3, 0, 0])],
                         [self.segment(end_quat=[0, 0, .70710678, .70710678])]):
            selector = self.selector()
            self.call(selector, 0, 1)
            _, report = self.call(selector, 1, 2, segments=segments)
            self.assertEqual(report["failure_count"], 0)

    def test_same_segment_counted_once(self):
        selector = self.selector()
        self.call(selector, 0, 1)
        self.call(selector, 1, 2, segments=[self.segment()])
        _, report = self.call(selector, 2, 3, segments=[self.segment()])
        self.assertEqual(report["failure_count"], 1)

    def test_world_direction_survives_rotation(self):
        selector = self.selector()
        self.call(selector, 0, 1)
        _, report = self.call(selector, 1, 2, segments=[self.segment()], quat=[0, 0, 1, 0])
        # Local backward now points world +X, the actual failed direction.
        self.assertEqual(report["failure_cost"][0], 0)
        self.assertGreater(report["failure_cost"][1], 0)

    def test_locality_expiry_reset_and_environment_isolation(self):
        selector = self.selector()
        self.call(selector, 0, 1)
        self.call(selector, 1, 2, segments=[self.segment()])
        _, other = self.call(selector, 1, 3, env=1)
        self.assertEqual(other["failure_count"], 0)
        _, far = self.call(selector, 2, 4, pos=[2, 0, 0])
        self.assertEqual(far["failure_cost"], [0., 0., 0.])
        _, expired = self.call(selector, 40, 5)
        self.assertEqual(expired["failure_count"], 0)
        episode = expired["episode"]
        selector.reset_env(0)
        _, reset = self.call(selector, 0, 1, stuck=False)
        self.assertEqual(reset["state"], "NORMAL")
        self.assertNotEqual(reset["episode"], episode)

    def test_cooldown_and_normal_scene_q_selection(self):
        selector = self.selector()
        self.call(selector, 0, 1)
        _, report = self.call(selector, 2, 2, stuck=False, pos=[1, 0, 0])
        self.assertEqual(report["state"], "COOLDOWN")
        _, report = self.call(selector, 2.5, 3, stuck=True, pos=[1, 0, 0])
        self.assertEqual(report["state"], "COOLDOWN")
        selected, report = self.call(selector, 4, 4, stuck=False, pos=[1, 0, 0])
        self.assertEqual(report["state"], "NORMAL")
        self.assertEqual(selected, 0)

    def test_ablation_has_no_memory_penalty(self):
        selector = self.selector("state_only")
        self.call(selector, 0, 1)
        selected, report = self.call(selector, 1, 2, segments=[self.segment()])
        self.assertEqual(selected, 0)
        self.assertEqual(report["failure_cost"], [0., 0., 0.])

    def test_missing_protocol_falls_back_and_invalid_data_rejected(self):
        selector = self.selector()
        selected, report = selector.select(0, self.paths, self.q, self.origin, self.quat, True)
        self.assertEqual(selected, 0)
        self.assertIn("fallback", report)
        self.call(selector, 2, 1)
        with self.assertRaises(ValueError):
            self.call(selector, 1, 2)
        with self.assertRaises(ValueError):
            RecoveryConfig(memory_ttl_s=0)
        with self.assertRaises(ValueError):
            selector.select(0, self.paths, [float('nan')]*3, None, None, False)

    def test_execution_tracks_only_applied_controls_and_copies_snapshots(self):
        tracker = ExecutionHistory(2)
        pos = np.zeros((2, 3)); quat = np.tile(self.quat, (2, 1))
        tracker.record(None, .1, pos, quat, pos, quat, [False, False])
        self.assertEqual(tracker.snapshot(), [[], []])
        tracker.record(4, .1, pos, quat, pos, quat, [False, False])
        old = tracker.snapshot()
        tracker.record(4, .1, pos, quat, pos, quat, [False, True])
        self.assertEqual(old[0][0]["duration_s"], .1)
        self.assertEqual(tracker.snapshot()[0][0]["duration_s"], .2)
        self.assertEqual(tracker.snapshot()[1], [])
        tracker.record(5, .1, pos, quat, pos, quat, [False, False])
        self.assertEqual([s["plan_id"] for s in tracker.snapshot()[0]], [4, 5])


if __name__ == '__main__':
    unittest.main()
