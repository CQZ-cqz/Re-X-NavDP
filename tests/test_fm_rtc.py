"""RTC CPU math/interface tests; no simulator or GPU usage."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import unittest
from unittest.mock import patch
import numpy as np
import torch
from FM_distillation.core.rtc import sample_with_rtc, fm_rtc_agent_class
from FM_distillation.core.evaluation import metric_summary
from FM_distillation.core.flow_generator import CompactFlowGenerator
from test_flow_generator import teacher_fixture


class RTCTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = CompactFlowGenerator(teacher_fixture(),depth=1).eval().requires_grad_(False)
        self.goal,self.rgbd = torch.randn(1,1,32),torch.randn(1,16,32)
        self.noise = torch.randn(1,8,24,3)
        self.kw = dict(prev_action=torch.zeros(1,24,3),valid_segment_len=[16],
                       guidance_factor=[.5]*6+[.05]*2,initial_noise=self.noise)

    def test_no_history_and_stuck_are_exact_rtc_off(self):
        expected = self.model.sample(self.goal,self.rgbd,initial_noise=self.noise)
        for change in (dict(valid_segment_len=[0]),dict(guidance_factor=0)):
            actual = sample_with_rtc(self.model,self.goal,self.rgbd,**{**self.kw,**change})
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_actual_transformer_under_outer_no_grad(self):
        stats = {}
        with torch.no_grad():
            actual = sample_with_rtc(self.model,self.goal,self.rgbd,**self.kw,stats=stats)
        self.assertEqual(actual.shape,(1,8,24,3))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertFalse(actual.requires_grad)
        self.assertEqual(stats['guided_steps'],2)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_guidance_moves_toward_prefix_not_away(self):
        # Zero velocity => clean estimate x, Jacobian I, analytically contract XY.
        noise = torch.ones_like(self.noise)
        with patch.object(self.model,'forward',side_effect=lambda x,*a: x*0):
            actual = sample_with_rtc(self.model,self.goal,self.rgbd,
                **{**self.kw,'initial_noise':noise})
        # Two updates with coefficients 2 and 10/3, dt=1/4, factor=.5.
        expected = (1-.25*2*.5)*(1-.25*(10/3)*.5)
        self.assertAlmostEqual(actual[0,0,0,0].item(),expected,places=6)
        self.assertLess(actual[0,0,0,0],actual[0,7,0,0])
        torch.testing.assert_close(actual[...,2],noise[...,2])
        torch.testing.assert_close(actual[:,:,16:,:],noise[:,:,16:,:])

    def test_rejects_bad_history_and_factors(self):
        for change in (dict(valid_segment_len=[25]),dict(guidance_factor=[1,2]),dict(beta=0)):
            with self.assertRaises(ValueError):
                sample_with_rtc(self.model,self.goal,self.rgbd,**{**self.kw,**change})

    def test_ne_uses_all_episodes_not_only_successes(self):
        rows = [dict(ne='0.2',success='1',spl='.8'),dict(ne='2.8',success='0',spl='0')]
        report = metric_summary(rows)
        self.assertEqual(report['ne_mean_m'],1.5)
        self.assertEqual(report['success_rate'],.5)
        self.assertEqual(report['ne_failure_mean_m'],2.8)

    def test_agent_passes_real_guidance_and_preserves_ranking_interface(self):
        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rtc_enabled,self.device = True,'cpu'
                self.point_encoder = torch.nn.Linear(3,32)
                self.rgbd_encoder = lambda *args: torch.zeros(1,16,32)
        class Agent:
            def __init__(self):
                self.navi_former,self.device = Teacher(),'cpu'
        def rank(teacher,student,goal,rgbd,embodiment,**kw):
            raw = student.sample(goal,rgbd,embodiment,**kw)
            return dict(trajectories=raw,scores=torch.zeros(1,8),top_trajectories=raw[:,:2])
        with patch('eval.src.policy_agent.NavDP_Agent',Agent), \
             patch('FM_distillation.core.flow_generator.CompactFlowGenerator',return_value=self.model), \
             patch('FM_distillation.core.flow_generator.generate_and_rank',side_effect=rank), \
             patch('torch.load',return_value=dict(signature=dict(teacher_sha256='t'),student=self.model.state_dict())):
            agent = fm_rtc_agent_class('unused','t')()
            result = agent.navi_former.predict_pointgoal_action_with_guidance(np.zeros((1,3)),None,None,
                prev_action=np.zeros((1,24,3)),valid_segment_len=[16],guidance_factor=np.ones((1,8))*.5,
                start_index=0,end_index=23,guidance_step=5,prefix_attention_schedule='exp')
        self.assertEqual(result[0].shape,(1,8,24,3))


if __name__ == '__main__':
    unittest.main()
