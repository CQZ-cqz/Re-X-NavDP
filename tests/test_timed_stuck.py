
# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import unittest
import numpy as np
from bridge.recovery.stuck import TimedStuckDetector


class TimedStuckTests(unittest.TestCase):
    def test_same_motion_at_different_planning_rates(self):
        for dt in (.05, .1, .25, .5):
            for speed, expected in ((0., True), (.1, True), (.2, False)):
                detector = TimedStuckDetector(2., .25)
                for t in np.arange(0., 4.001, dt):
                    stuck, report = detector.update([speed*t, 0], t)
                    if t < 2.-1e-9:
                        self.assertFalse(stuck)
                self.assertEqual(stuck, expected)
                self.assertAlmostEqual(report['max_displacement_m'], 2*speed)

    def test_duplicate_timestamps_do_not_fill_window(self):
        detector = TimedStuckDetector()
        for _ in range(100):
            self.assertFalse(detector.update([0, 0], 1.)[0])
        self.assertEqual(len(detector.history), 1)

    def test_reset_rewind_gap_and_missing_time(self):
        for transition in ('reset', 'rewind', 'gap', 'missing', 'clock'):
            detector = TimedStuckDetector()
            for t in (0., 1., 2.):
                stuck, _ = detector.update([0, 0], t)
            self.assertTrue(stuck)
            if transition == 'reset':
                detector.reset()
                args = ([0, 0], 2.)
            elif transition == 'rewind':
                args = ([0, 0], 0.)
            elif transition == 'gap':
                args = ([0, 0], 6.)
            elif transition == 'missing':
                args = ([0, 0], None)
            else:
                args = ([0, 0], 2., 'monotonic')
            self.assertFalse(detector.update(*args)[0])

    def test_boundary_interpolation(self):
        detector = TimedStuckDetector()
        for t in (0., .7, 1.4, 2.1):
            stuck, report = detector.update([.2*t, 0], t)
        self.assertFalse(stuck)
        self.assertAlmostEqual(report['max_displacement_m'], .4)

    def test_return_to_start_is_not_stationary(self):
        detector = TimedStuckDetector()
        for t, x in ((0, 0), (1, 1), (2, 0)):
            stuck, _ = detector.update([x, 0], t)
        self.assertFalse(stuck)


if __name__ == '__main__':
    unittest.main()
