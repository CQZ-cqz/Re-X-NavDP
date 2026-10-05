"""CPU numerical and deployment-path checks, without loading vision weights."""

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
from diffusers import DDIMScheduler, DDPMScheduler

from ddim.core.diffusion_sampling import ddim_step, sampling_timesteps
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment


class DDIMTests(unittest.TestCase):
    def test_matches_diffusers_uniform_grid(self):
        ref = DDIMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
                            prediction_type="epsilon", clip_sample=True)
        for steps in (10, 5):
            ref.set_timesteps(steps)
            x, eps = torch.randn(2, 24, 3), torch.randn(2, 24, 3)
            for index, t in enumerate(ref.timesteps):
                next_t = int(ref.timesteps[index + 1]) if index + 1 < steps else -1
                actual, clean = ddim_step(x, eps, int(t), next_t, ref.alphas_cumprod)
                expected = ref.step(eps, t, x, eta=0)
                torch.testing.assert_close(actual, expected.prev_sample)
                torch.testing.assert_close(clean, expected.pred_original_sample)

    def test_nonuniform_grid_oracle_and_rng(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2")
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.double()
        clean = torch.randn(2, 24, 3, dtype=torch.float64) * .1
        eps = torch.randn_like(clean)
        for steps in (10, 5, 4, 2):
            grid = sampling_timesteps("ddim", steps)
            self.assertEqual((int(grid[0]), int(grid[-1])), (9, 0))
            self.assertTrue(bool((grid[:-1] > grid[1:]).all()))
            a = scheduler.alphas_cumprod[9]
            x = a.sqrt() * clean + (1-a).sqrt() * eps
            rng = torch.get_rng_state().clone()
            for index, t in enumerate(grid):
                next_t = int(grid[index + 1]) if index + 1 < steps else -1
                x, _ = ddim_step(x, eps, int(t), next_t, scheduler.alphas_cumprod, clip_sample=False)
                a_next = scheduler.alphas_cumprod[next_t] if next_t >= 0 else torch.tensor(1.)
                torch.testing.assert_close(x, a_next.sqrt()*clean + (1-a_next).sqrt()*eps,
                                           atol=1e-5, rtol=1e-4)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_options(self):
        for args in (("ddpm", 5, 0), ("ddim", 1, 0), ("ddim", 11, 0),
                     ("ddim", 4, float("nan")), ("ddim", 4, -1)):
            with self.assertRaises(ValueError):
                sampling_timesteps(*args)

    def test_real_sampling_loop_routing_and_rtc(self):
        # Construct a lightweight instance; retain the real sampling/RTC code.
        for rtc in (False, True):
            net = NavDP_Policy_Embodiment.__new__(NavDP_Policy_Embodiment)
            torch.nn.Module.__init__(net)
            net.device, net.predict_size, net.ft_step = "cpu", 24, 6
            net.sampler, net.ddim_eta, net.rtc_enabled = "ddim", 0., rtc
            net.sampling_timesteps = sampling_timesteps("ddim", 4)
            net.noise_scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2")
            net.rgbd_encoder = lambda *args: torch.zeros(1, 128, 8)
            net.point_encoder = torch.nn.Linear(3, 8)
            net.weight = np.ones(25)
            calls = []
            def noise(branch):
                def forward(x, t, *args):
                    calls.append((branch, int(t)))
                    return x * .1
                return forward
            net.predict_noise = noise("base")
            net.predict_noise_ft = noise("ft")
            net.smooth_trajectory = lambda x, **kw: x
            net.smooth_cumulative_trajectory = lambda x, **kw: x
            net.predict_pointgoal_q = lambda path, *args, **kw: (path.sum((1, 2)), path.sum((1, 2)))
            with patch.object(net, "pinv_corrected_velocity", wraps=net.pinv_corrected_velocity) as guide:
                paths, scores, best, _ = net.predict_pointgoal_action_with_guidance(
                    np.zeros((1, 3)), None, None, 8, np.array([6]), np.zeros((1, 24, 3)),
                    0, 6, np.full((1, 8), .05), guidance_step=5, embodiment=1)
                self.assertEqual(guide.call_count, 2 if rtc else 0)
            self.assertEqual(calls, [("base", 9), ("base", 6), ("ft", 3), ("ft", 0)])
            self.assertEqual(paths.shape, (1, 8, 24, 3))
            self.assertEqual(best.shape, (1, 2, 24, 3))
            self.assertTrue(np.isfinite(paths).all() and np.isfinite(scores).all())

    def test_initial_noise_pairs_samplers(self):
        # Fixed initial noise must make DDIM (eta=0, no RTC) fully deterministic,
        # while DDPM still draws per-step scheduler noise. This is what
        # compare_ddim_ddpm.py relies on for the paired-noise comparison.
        for sampler in ("ddim", "ddpm"):
            net = NavDP_Policy_Embodiment.__new__(NavDP_Policy_Embodiment)
            torch.nn.Module.__init__(net)
            net.device, net.predict_size, net.ft_step = "cpu", 24, 6
            net.sampler, net.ddim_eta, net.rtc_enabled = sampler, 0., False
            steps = 10 if sampler == "ddpm" else 4
            net.sampling_timesteps = sampling_timesteps(sampler, steps)
            net.noise_scheduler = DDPMScheduler(
                num_train_timesteps=10, beta_schedule="squaredcos_cap_v2")
            net.rgbd_encoder = lambda *args: torch.zeros(1, 128, 8)
            net.point_encoder = torch.nn.Linear(3, 8)
            net.weight = np.ones(25)
            net.predict_noise = lambda x, t, *args: x * .1
            net.predict_noise_ft = lambda x, t, *args: x * .1
            net.smooth_trajectory = lambda x, **kw: x
            net.smooth_cumulative_trajectory = lambda x, **kw: x
            net.predict_pointgoal_q = lambda path, *args, **kw: (path.sum((1, 2)), path.sum((1, 2)))
            noise = torch.randn(8, 24, 3)

            def run(seed):
                torch.manual_seed(seed)
                paths, _, _, _ = net.predict_pointgoal_action_with_guidance(
                    np.zeros((1, 3)), None, None, 8, np.array([6]), np.zeros((1, 24, 3)),
                    0, 6, np.full((1, 8), .05), guidance_step=5, embodiment=1,
                    initial_noise=noise)
                return paths

            if sampler == "ddim":
                np.testing.assert_allclose(run(0), run(12345))
            else:
                self.assertFalse(np.allclose(run(0), run(12345)))


if __name__ == "__main__":
    unittest.main()
