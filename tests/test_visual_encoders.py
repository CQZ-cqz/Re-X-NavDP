
# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import tempfile
from dataclasses import asdict
import time
import unittest
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from rl.src.encoder import (
    DAV2_TYPE,
    MetricDepthPreprocessor,
    YOLO26DepthRGBDEncoder,
    build_rgbd_encoder,
    resolve_policy_visual_config,
)
from rl.src.policy import PolicyConfig, ReactiveActorCritic
from rl.src.runtime import LatestRGBDWorker
from rl.src.runner import validate_checkpoint, ACTION_MAPPING_RESIDUAL
from rl.src.observation import STATE_VERSION
from experiments.smoke.smoke_env import SmokeVectorEnv


class _Indexed(nn.Module):
    def __init__(self, index, operation):
        super().__init__()
        self.i, self.f, self.operation = index, -1, operation

    def forward(self, value):
        return self.operation(value)


class Depth(nn.Module):
    """Fake final head; class name and ``f`` match the Ultralytics graph API."""
    def __init__(self, index=17):
        super().__init__()
        self.i, self.f = index, [16]

    def forward(self, value):  # pragma: no cover - early exit must make this unreachable
        raise AssertionError("Depth head must not execute during P3 extraction")


class _FakeYOLOCore(nn.Module):
    def __init__(self):
        super().__init__()
        operations = [
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.Conv2d(16, 64, 3, 2, 1),
        ] + [nn.Identity() for _ in range(14)]
        self.model = nn.ModuleList([_Indexed(i, op) for i, op in enumerate(operations)] + [Depth()])
        self.save = [16]


def _encoder(training=False):
    core = _FakeYOLOCore()
    return YOLO26DepthRGBDEncoder(core, deepcopy(core), fingerprint="fake-yolo",
        depth_preprocessor=MetricDepthPreprocessor(training=training))


class VisualEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(4)
        self.rgb = np.random.default_rng(3).integers(0, 256, (2, 97, 131, 3), dtype=np.uint8)
        self.depth = np.ones((2, 97, 131, 1), np.float32)

    def test_yolo_p3_shape_adapter_gradient_and_frozen_backbone(self):
        encoder = _encoder()
        features = encoder(self.rgb, self.depth)
        self.assertEqual(encoder.metadata["visual_feature_layer"], "model.16:Depth.P3/stride8")
        self.assertEqual(encoder.metadata["raw_feature_shape"], (64, 28, 28))
        self.assertEqual(features["rgb_tokens"].shape, (2, 64, 64))
        self.assertEqual(features["depth_tokens"].shape, (2, 64, 64))
        # This is the trainable pool-then-Linear adapter described in the design.
        adapter = nn.Linear(64, 128)
        visual = torch.cat((adapter(features["rgb_tokens"]), adapter(features["depth_tokens"])), 1)
        self.assertEqual(visual.shape, (2, 128, 128))
        visual.square().mean().backward()
        self.assertGreater(adapter.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        encoder.train()
        self.assertFalse(encoder.training)

    def test_policy_dimension_is_resolved_and_policy_adapter_trains(self):
        encoder = _encoder()
        features = encoder(self.rgb, self.depth)
        options = resolve_policy_visual_config(dict(state_dim=28), encoder)
        self.assertEqual(options["token_dim"], 64)
        policy = ReactiveActorCritic(PolicyConfig(**options))
        obs = SmokeVectorEnv(features, policy.config).reset()
        policy.evaluate(obs).square().mean().backward()
        self.assertGreater(policy.fusion.rgb_project[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(policy.fusion.depth_project[1].weight.grad.abs().sum().item(), 0)

    def test_invalid_depth_is_finite_and_eval_is_deterministic(self):
        encoder = _encoder()
        self.depth[0, 0, :5, 0] = [np.nan, np.inf, -1., 0., 10.]
        first, second = encoder(self.rgb, self.depth), encoder(self.rgb, self.depth)
        for key in ("rgb_tokens", "depth_tokens", "depth_valid_fraction"):
            self.assertTrue(torch.isfinite(first[key]).all())
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_latest_feature_cache_does_not_reencode_on_snapshot(self):
        encoder = _encoder()
        worker = LatestRGBDWorker(encoder)
        try:
            worker.submit(self.rgb, self.depth, 1, .1, np.zeros((2, 3)),
                np.tile([0., 0., 0., 1.], (2, 1)))
            deadline = time.monotonic()+3
            while worker.snapshot() is None and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertIsNotNone(worker.snapshot())
            count = encoder.forward_count
            for _ in range(5):
                self.assertIsNotNone(worker.snapshot())
            self.assertEqual(encoder.forward_count, count)
        finally:
            worker.close()

    def test_checkpoint_rejects_visual_backend_mismatch(self):
        encoder = _encoder()
        config = PolicyConfig(token_dim=encoder.token_dim)
        checkpoint = {"format_version": 1, "policy_config": asdict(config),
            "state_version": STATE_VERSION, "action_mapping_version": ACTION_MAPPING_RESIDUAL,
            "preprocess_version": encoder.preprocess_version,
            "encoder_fingerprint": encoder.fingerprint,
            "visual_encoder_metadata": encoder.metadata}
        validate_checkpoint(checkpoint, config, encoder.metadata)
        wrong = dict(encoder.metadata, visual_encoder_type=DAV2_TYPE)
        with self.assertRaisesRegex(ValueError, "Visual encoder mismatch"):
            validate_checkpoint(checkpoint, config, wrong)

    def test_factory_missing_weights_fails_without_network_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "YOLO26-Depth weights not found"):
                build_rgbd_encoder({"type": "yolo26_depth", "weights": directory+"/missing.pt"})
        self.assertEqual(DAV2_TYPE, "depth_anything_v2")


if __name__ == "__main__":
    unittest.main()
