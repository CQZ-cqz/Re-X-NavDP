"""CPU-only interface tests for the self-contained deploy checkpoint path.

The deploy.pt bundles a frozen teacher backbone + distilled student. These
tests verify the ``NavDP_Agent`` hook and the ranked-planning return signature
without loading vision weights or touching the GPU.
"""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

from FM_distillation.core.fm_backbone import fm_deploy_agent_class, DeployPolicy


class DeployAgentTests(unittest.TestCase):
    def test_factory_builds_navi_former_from_deploy_path(self):
        class Agent:
            def __init__(self):
                self.device = "cpu"

        class FakeDeploy:
            def __init__(self, checkpoint, device="cuda:0", rtc_enabled=False, beta=5.0):
                self.checkpoint, self.device = checkpoint, device
                self.rtc_enabled, self.beta = rtc_enabled, beta

        with patch("eval.src.policy_agent.NavDP_Agent", Agent), \
             patch("FM_distillation.core.fm_backbone.DeployPolicy", FakeDeploy):
            cls = fm_deploy_agent_class("/tmp/deploy.pt", rtc_enabled=True, beta=7.0)
            navi = cls()._build_navi_former(navi_model="ignored-posttrain", **{})
        self.assertIsInstance(navi, FakeDeploy)
        self.assertEqual(navi.checkpoint, "/tmp/deploy.pt")
        self.assertTrue(navi.rtc_enabled)
        self.assertEqual(navi.beta, 7.0)

    def test_predict_returns_ranked_numpy_tuple(self):
        policy = DeployPolicy.__new__(DeployPolicy)
        policy.rtc_enabled = False
        policy.beta = 5.0
        policy.backbone = SimpleNamespace(
            device="cpu",
            point_encoder=torch.nn.Linear(3, 8),
            rgbd_encoder=lambda img, dep: torch.zeros(1, 16, 8),
        )
        policy.student = SimpleNamespace()
        policy._generate_and_rank = lambda network, sampler, goal, rgbd, embodiment, candidates, steps: dict(
            trajectories=torch.zeros(1, 8, 24, 3),
            scores=torch.zeros(1, 8),
            top_trajectories=torch.zeros(1, 2, 24, 3),
        )
        out = policy.predict_pointgoal_action_with_guidance(
            np.zeros((1, 3)), None, None, sample_num=8)
        self.assertEqual(out[0].shape, (1, 8, 24, 3))
        self.assertEqual(out[1].shape, (1, 8))
        self.assertEqual(out[2].shape, (1, 2, 24, 3))
        self.assertIsNone(out[3])

    def test_predict_rejects_non_eight_candidates(self):
        policy = DeployPolicy.__new__(DeployPolicy)
        policy.rtc_enabled = False
        policy.backbone = SimpleNamespace(device="cpu")
        with self.assertRaisesRegex(ValueError, "eight candidates"):
            policy.predict_pointgoal_action_with_guidance(
                np.zeros((1, 3)), None, None, sample_num=4)

    def test_labeled_mixed_data_delegates_to_loader(self):
        from FM_distillation.core.training import labeled_mixed_data
        with patch("FM_distillation.core.merging.load_mixed_dataset",
                   return_value=("snap", "groups", "cache")):
            self.assertEqual(labeled_mixed_data("/tmp/mixed"), ("snap", "groups", "cache"))


if __name__ == "__main__":
    unittest.main()
