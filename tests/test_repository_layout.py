"""Release boundaries, independent of GPU and external data."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_baseline_scripts_are_only_upstream_entries(self):
        actual = {p.name for p in (ROOT/'baselines/x-navdp/scripts').iterdir() if p.is_file()}
        self.assertEqual(actual, {'aggregate_success.py', 'run_ddp_train.sh'})

    def test_no_downstream_compatibility_trees(self):
        for relative in ('baselines/x-navdp/src/reactive', 'baselines/x-navdp/src/recovery', 'baselines/x-navdp/development',
                         'baselines/x-navdp/eval/src/flow_generator.py', 'docs/legacy-development'):
            self.assertFalse((ROOT/relative).exists(), relative)
        for relative in ('rl/src/policy.py', 'FM_distillation/src/flow_generator.py',
                         'ddim/src/diffusion_sampling.py', 'bridge/planner_executor_bridge.py'):
            self.assertTrue((ROOT/relative).is_file(), relative)


if __name__ == '__main__':
    unittest.main()
