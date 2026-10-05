"""DDIM epsilon updates on explicit original-training-time grids.

Unlike a fixed-stride scheduler, each update uses the actual next selected
timestep. This matters for 10 training steps sampled with 4 inference steps.
"""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import math

import torch


def sampling_timesteps(sampler="ddpm", steps=10, eta=0.0):
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler must be ddpm or ddim")
    if not isinstance(steps, int) or not 2 <= steps <= 10:
        raise ValueError("inference steps must be an integer between 2 and 10")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("DDIM eta must be between 0 and 1")
    if sampler == "ddpm" and (steps != 10 or eta != 0):
        raise ValueError("DDPM baseline requires 10 steps and eta=0")
    return torch.linspace(9, 0, steps).round().long()


def ddim_step(sample, epsilon, timestep, next_timestep, alphas_cumprod,
              eta=0.0, clip_sample=True, clip_sample_range=1.0):
    """DDIM Eq. 12/16; returns (next noisy sample, predicted clean sample).

    next_timestep=-1 denotes clean output (alpha=1). Epsilon is not recomputed
    after clipping, matching diffusers' default use_clipped_model_output=False.
    """
    alpha = alphas_cumprod[timestep].to(sample)
    alpha_next = (alphas_cumprod[next_timestep].to(sample)
                  if next_timestep >= 0 else sample.new_tensor(1.0))
    clean = (sample - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
    if clip_sample:
        clean = clean.clamp(-clip_sample_range, clip_sample_range)
    variance = ((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)).clamp(min=0)
    sigma = eta * variance.sqrt()
    previous = alpha_next.sqrt() * clean + (1 - alpha_next - sigma.square()).clamp(min=0).sqrt() * epsilon
    if eta > 0 and next_timestep >= 0:
        previous = previous + sigma * torch.randn_like(sample)
    return previous, clean
