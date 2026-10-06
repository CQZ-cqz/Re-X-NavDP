"""CPU tests: conditioning parity, FM math, candidate ordering, frozen teacher."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from FM_distillation.src.flow_generator import CompactFlowGenerator, generate_and_rank
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment


class Position(nn.Module):
    def __init__(self, width, length):
        super().__init__()
        self.embedding = nn.Embedding(length, width)

    def forward(self, x):
        return self.embedding(torch.arange(x.shape[1], device=x.device))[None]


class Time(nn.Module):
    def forward(self, t):
        phase = t[:, None] * torch.exp(torch.arange(16, device=t.device)*(-math.log(10000)/15))
        return torch.cat([phase.sin(), phase.cos()], -1)


def teacher_fixture():
    teacher = NavDP_Policy_Embodiment.__new__(NavDP_Policy_Embodiment)
    nn.Module.__init__(teacher)
    teacher.token_dim, teacher.memory_size, teacher.predict_size = 32, 1, 24
    teacher.attention_heads, teacher.distinguish_embodiment, teacher.device = 4, True, "cpu"
    teacher.point_encoder = nn.Linear(3, 32)
    teacher.cond_pos_embed, teacher.out_pos_embed = Position(32, 20), Position(32, 24)
    teacher.time_emb = Time()
    teacher.embodiment_embedding = nn.Embedding(3, 32)
    teacher.embody_tgt_delta = nn.Linear(32, 32)
    teacher.embody_out_film = nn.Linear(32, 64)
    teacher.tgt_mask = torch.triu(torch.full((24, 24), -torch.inf), diagonal=1)
    teacher.weight = np.ones(25)
    return teacher.eval()


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.teacher = teacher_fixture()
        self.student = CompactFlowGenerator(self.teacher, depth=2)
        self.goal, self.rgbd = torch.randn(2, 1, 32), torch.randn(2, 16, 32)

    def test_loss_gradients_and_frozen_conditions(self):
        self.goal.requires_grad_()
        self.rgbd.requires_grad_()
        target = torch.randn(2, 24, 3, requires_grad=True)
        loss = self.student.flow_loss(target, self.goal, self.rgbd, torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.student.velocity_head.weight.grad.abs().sum(), 0)
        self.assertIsNone(target.grad)
        self.assertIsNone(self.goal.grad)
        self.assertIsNone(self.rgbd.grad)
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        for name in self.student.condition_names:
            self.assertTrue(all(not p.requires_grad for p in getattr(self.student, name).parameters()))

    def test_exact_condition_injection_parity(self):
        student = self.student.eval()
        # Same action projection/decoder/output head isolate condition plumbing.
        teacher = self.teacher
        teacher.input_embed_ft = student.input_embed
        teacher.decoder_ft = student.decoder
        teacher.layernorm_ft = student.layernorm
        teacher.action_head_ft = student.velocity_head
        x, t, embodiment = torch.randn(2, 24, 3), torch.tensor([.2, .7]), torch.tensor([0, 2])
        actual = student(x, t, self.goal, self.rgbd, embodiment)
        expected = teacher.predict_noise_ft(x, t*9, self.goal, self.rgbd, embodiment)
        torch.testing.assert_close(actual, expected)

    def test_linear_flow_loss_oracle_and_euler(self):
        target, noise = torch.randn(2, 24, 3), torch.randn(2, 24, 3)
        def oracle(x, t, *args):
            torch.testing.assert_close(x, (1-t[:, None, None])*noise+t[:, None, None]*target)
            return target-noise
        with patch.object(self.student, "forward", side_effect=oracle):
            loss = self.student.flow_loss(target, self.goal, self.rgbd,
                                          noise=noise, t=torch.tensor([0., 1.]))
            self.assertEqual(loss.item(), 0)
        self.student.eval()
        initial = torch.randn(2, 3, 24, 3)
        with patch.object(self.student, "forward", side_effect=lambda x, *args: torch.ones_like(x)*2):
            for steps in (1, 2, 4):
                out = self.student.sample(self.goal, self.rgbd, candidates=3, steps=steps, initial_noise=initial)
                torch.testing.assert_close(out, initial+2)

    def test_candidates_repeatability_and_validation(self):
        self.student.eval()
        initial = torch.randn(2, 8, 24, 3)
        a = self.student.sample(self.goal, self.rgbd, torch.tensor([0, 2]), initial_noise=initial)
        b = self.student.sample(self.goal, self.rgbd, torch.tensor([0, 2]), initial_noise=initial)
        torch.testing.assert_close(a, b)
        self.assertEqual(a.shape, (2, 8, 24, 3))
        self.assertFalse(torch.equal(a[:, 0], a[:, 1]))
        for kwargs in ({"steps": 0}, {"candidates": 0}, {"embodiment": 3}):
            with self.assertRaises(ValueError):
                self.student.sample(self.goal, self.rgbd, **kwargs)
        self.student.train()
        with self.assertRaises(RuntimeError):
            self.student.sample(self.goal, self.rgbd)

    def test_ranking_uses_teacher_coordinate_and_q_convention(self):
        self.student.eval()
        raw = torch.randn(2, 8, 24, 3)*.1
        captured = {}
        def q(path, rgbd, goal, **kwargs):
            captured["path"], captured["embodiment"] = path, kwargs["embodiment"]
            scores = torch.arange(16, dtype=path.dtype)
            return scores, scores+2
        with patch.object(self.student, "sample", return_value=raw), patch.object(
                self.teacher, "predict_pointgoal_q", side_effect=q):
            out = generate_and_rank(self.teacher, self.student, self.goal, self.rgbd, torch.tensor([0, 2]))
        flat = raw.reshape(16, 24, 3)
        expected = self.teacher.smooth_cumulative_trajectory(
            torch.cumsum(flat/4, 1), num_points=25, smooth_factor=.5, weight=self.teacher.weight)
        torch.testing.assert_close(captured["path"], expected)
        torch.testing.assert_close(captured["embodiment"], torch.tensor([0]*8+[2]*8))
        torch.testing.assert_close(out["scores"], torch.arange(16).reshape(2, 8).float()+1)
        torch.testing.assert_close(out["top_indices"], torch.tensor([[7, 6], [7, 6]]))
        self.assertEqual(out["top_trajectories"].shape, (2, 2, 24, 3))


if __name__ == "__main__":
    unittest.main()
